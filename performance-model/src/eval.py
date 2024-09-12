"""
Script to evaluate the model
"""

import os
from src.utils.opt import Opts

from src.model import MODEL_REGISTRY
import lightning.pytorch as pl
from lightning.pytorch.trainer import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

import datetime
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb.Table.MAX_ARTIFACTS_ROWS = 1000000


def check(config):
    model_class = MODEL_REGISTRY.get(config.model["name"])
    model = model_class.load_from_checkpoint(
        config["global"]["checkpoint"], config=config, strict=True
    )
    # manually change config to add information about test stage
    model.cfg = config
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"eval-{config['global']['run_name']}-{time_str}"

    wandb_logger = WandbLogger(
        project=config["global"]["project_name"],
        name=run_name,
        save_dir=config["global"]["save_dir"],
        entity=config["global"]["username"],
    )
    # only save on rank-0 process if run on multiple GPUs
    # https://github.com/Lightning-AI/lightning/issues/13166
    if rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(config)

    trainer = pl.Trainer(
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        logger=wandb_logger,
        precision=config["trainer"]["precision"],
    )

    trainer.test(model)
    del trainer
    del config
    del model


if __name__ == "__main__":
    config = Opts(cfg="configs/eval/config_cldrive_v1.yaml").parse_args()
    seed_everything(config["global"]["SEED"])
    check(config)
