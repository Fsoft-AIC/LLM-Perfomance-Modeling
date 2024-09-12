from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
import os
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
        self.scale = cfg["model"]["output_scale"]
        self.mse_loss_func = nn.MSELoss(reduction="mean")

        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(**self.cfg["tokenizer"]["args"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.lora_config = None
        self.bnb_config = None

        if 'lora' in cfg:
            from peft import LoraModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
            self.lora_config = LoraConfig(task_type="CAUSAL_LM",
                                        r=cfg['lora']['r'],
                                        lora_alpha=cfg['lora']['alpha'],
                                        target_modules=cfg['lora']['layers'],
                                        lora_dropout=cfg['lora']['dropout'], )
            if cfg['lora']['qlora']:
                self.bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                                    bnb_4bit_use_double_quant=True,
                                                        bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16)
        if self.bnb_config is not None:
        # setup core model
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                self.cfg["model"]["base_model"],
                load_in_4bit=True, torch_dtype=torch.bfloat16,
                quantization_config=self.bnb_config,
                device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}, # compible with huggingface accelerator
                max_memory = {i: '81900MB' for i in range(torch.cuda.device_count())}
            )
        else: 
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                self.cfg["model"]["base_model"],
            )
        model_config = pretrained_model.config
        
        model_config.max_seq_len = self.cfg["model"]["max_seq_len"]  # self-defined
        self.model = CodeGenForPerfModeling(model_config)
        # print(self.model)
        # raise NotImplementedError
        self.model.transformer = pretrained_model.transformer
        self.model.lm_head = pretrained_model.lm_head
        if self.lora_config is not None:
            if self.bnb_config is not None:
                self.model = prepare_model_for_kbit_training(self.model, False)
            self.model = get_peft_model(self.model, self.lora_config, "default")

        self.save_hyperparameters()

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
            self.test_result = {"pred": [], "target": [], "prompt": []}

    def forward(self, batch):
        # since batch contains input of model and information for metric, we need to
        # pass only model's input to model
        input_cols = ["input_ids", "execution_time", "attention_mask"]

        batch_inputs = {key: batch[key] for key in input_cols}

        outputs = self.model(**batch_inputs)

        return outputs

    def compute_loss(self, forwarded_output, input_batch):
        """
        Function to compute loss
        Args:
            forwarded_output: output of `forward` method
            input_batch: input of batch method

        Returns:
            loss: computed loss
        """
        gt_execution_times = input_batch["execution_time"]
        # manually cast to half precision due to deepspeed limitation https://github.com/microsoft/DeepSpeed/issues/550
        pred_execution_times = forwarded_output["execution_time"].to(gt_execution_times.dtype)

        loss = self.mse_loss_func(pred_execution_times, gt_execution_times)

        return loss.to(dtype=gt_execution_times.dtype)

    def extract_target_from_batch(self, batch):
        # convert to float32 for numerical stability when computing metrics
        return self.scale_back(batch["execution_time"].detach().to('cpu', torch.float32))

    def extract_pred_from_forwarded_output(self, forwarded_output):
        # convert to float32 for numerical stability when computing metrics
        return self.scale_back(forwarded_output["execution_time"].detach().to('cpu', torch.float32))

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
        self.test_result["pred"].append(preds.detach().cpu().numpy())
        self.test_result["target"].append(targets.detach().cpu().numpy())
        self.test_result["prompt"].extend(batch["prompt"])

    def on_test_epoch_end(self):
        import pandas as pd

        # concatenate all test data
        preds = np.concatenate(self.test_result["pred"])
        targets = np.concatenate(self.test_result["target"])
        # save result to csv
        result = pd.DataFrame(
            {
                "pred": preds,
                "target": targets,
                "prompt": self.test_result["prompt"],
            }
        )
        if hasattr(self.logger, "log_table"):
            self.logger.log_table("result", dataframe=result)

        # reset for next epoch
        self.test_result["pred"].clear()
        self.test_result["target"].clear()
        self.test_result["prompt"].clear()

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

    def scale_back(self, arr):
        if self.scale == "log2":
            return 2**arr
        elif self.scale == "log10":
            return 10**arr
        elif self.scale == "original":
            return arr
