from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch import nn
from torch.utils.data import DataLoader
import torch
from torchmetrics import MetricCollection
from src.model import MODEL_REGISTRY
from src.dataset import DATASET_REGISTRY
from src.optimizer import OPTIMIZER_REGISTRY
from src.lrscheduler import LRSCHEDULER_REGISTRY
from src.metric import METRIC_REGISTRY
from src.utils.device import detach
from src.model.core.codegen import CodeGenForPerfModeling


@MODEL_REGISTRY.register()
class CodeGenLightningModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_metric = None
        self.val_metric = None
        self.test_metric = None
        self.tokenizer = None

        # setup tokenizer
        self.tokenizer = __setup_tokenizer(self.cfg["tokenizer"])

        # setup core model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model"]["base_model"]
        )
        self.save_hyperparameters()

    def __setup_tokenizer(self, config):
        tokenizer = AutoTokenizer.from_pretrained(**config["args"])
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def setup(self, stage):
        """
        Setup model before train/validate/test. Note that this method just setting up things
        that vary between stages, model's information are set in the __init__ method.
        """
        # setup metric
        metrics = [
            METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
            if mcfg["args"]
            else METRIC_REGISTRY.get(mcfg["name"])()
            for mcfg in self.cfg["metric"]
        ]
        metrics = MetricCollection({metric.name: metric for metric in metrics})

        # setup dataset
        if stage in ["fit", "validate"]:
            self.train_dataset = DATASET_REGISTRY.get(
                self.cfg["dataset"]["train"]["name"]
            )(
                config=self.cfg["dataset"]["train"]["args"],
                tokenizer=self.tokenizer,
            )
            self.val_dataset = DATASET_REGISTRY.get(self.cfg["dataset"]["val"]["name"])(
                config=self.cfg["dataset"]["val"]["args"],
                tokenizer=self.tokenizer,
            )
            self.train_metric = metrics.clone(prefix="train/")
            self.val_metric = metrics.clone(prefix="val/")
        elif stage == "test":
            self.test_dataset = DATASET_REGISTRY.get(
                self.cfg["dataset"]["test"]["name"]
            )(
                config=self.cfg["dataset"]["test"]["args"],
                tokenizer=self.tokenizer,
            )
            self.test_metric = metrics.clone(prefix="test/")
            self.test_step_preds = []
            self.test_step_targets = []
            self.test_prompts = []

    def forward(self, batch):
        # since batch contains input of model and information for metric, we need to
        # pass only model's input to model
        input_cols = ["input_ids", "attention_mask", "labels"]

        batch_inputs = {key: batch[key] for key in input_cols}

        output_dict = self.model(**batch_inputs)
        logits, loss = output_dict["logits"], output_dict["loss"]

        return dict(logits=logits, loss=loss)

    def compute_loss(self, forwarded_output, input_batch):
        """
        Function to compute loss
        Args:
            forwarded_output: output of `forward` method
            input_batch: input of batch method

        Returns:
            loss: computed loss
        """

        return forwarded_output["loss"]

    def extract_target_from_batch(self, batch):
        return self.scale_back(batch["execution_time"])

    def extract_pred_from_forwarded_output(self, forwarded_output):
        return self.scale_back(forwarded_output["execution_time"])

    def training_step(self, batch, batch_idx):
        # 1. get embeddings from model
        forwarded_output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_output=forwarded_output, input_batch=batch)
        # 3. Update metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        targets = self.extract_target_from_batch(batch)
        preds = self.extract_pred_from_forwarded_output(forwarded_output)
        output = self.train_metric(preds, targets)
        self.log_dict(output, on_step=True, on_epoch=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        forwarded_output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_output=forwarded_output, input_batch=batch)
        # 3. Update metric for each batch
        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        targets = self.extract_target_from_batch(batch)
        preds = self.extract_pred_from_forwarded_output(forwarded_output)
        output = self.val_metric(preds, targets)
        self.log_dict(output, on_step=True, on_epoch=True, sync_dist=True)

        return {"loss": detach(loss)}

    def test_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        forwarded_output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_output=forwarded_output, input_batch=batch)
        # 3. Update metric for each batch
        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        targets = self.extract_target_from_batch(batch)
        preds = self.extract_pred_from_forwarded_output(forwarded_output)
        output = self.test_metric(preds, targets)
        self.log_dict(output, on_step=True, on_epoch=True, sync_dist=True)

        # 4. record test data to accumulate and export to csv at end of epoch
        self.test_step_preds.append(preds.detach().cpu().numpy())
        self.test_step_targets.append(targets.detach().cpu().numpy())
        self.test_prompts.extend(batch["prompt"])

    def on_test_epoch_end(self):
        import pandas as pd

        # concatenate all test data
        preds = np.concatenate(self.test_step_preds)
        targets = np.concatenate(self.test_step_targets)
        # save result to csv
        result = pd.DataFrame(
            {
                "pred": preds,
                "target": targets,
                "prompt": self.test_prompts,
            }
        )
        if hasattr(self.logger, "log_table"):
            self.logger.log_table("result", dataframe=result)

        # reset for next epoch
        self.test_step_preds.clear()
        self.test_step_targets.clear()
        self.test_prompts.clear()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = DataLoader(
            dataset=self.train_dataset,
            **self.cfg["data_loader"]["train"]["args"],
        )
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loader = DataLoader(
            dataset=self.val_dataset,
            **self.cfg["data_loader"]["val"]["args"],
        )
        return val_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_loader = DataLoader(
            dataset=self.test_dataset,
            **self.cfg["data_loader"]["test"]["args"],
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = OPTIMIZER_REGISTRY.get(self.cfg["optimizer"]["name"])(
            self.parameters(), **self.cfg["optimizer"]["args"]
        )
        scheduler = LRSCHEDULER_REGISTRY.get(self.cfg["lr_scheduler"]["name"])(
            optimizer, self.cfg["lr_scheduler"]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
