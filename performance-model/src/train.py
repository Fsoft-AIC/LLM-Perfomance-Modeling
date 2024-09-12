import torch
from lightning.pytorch.trainer import seed_everything
import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import datetime
import os
import tempfile

from src.callback import CALLBACK_REGISTRY
from src.model import MODEL_REGISTRY
from src.utils.opt import Opts

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(config):
    model = MODEL_REGISTRY.get(config["model"]["name"])(config)

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['global']['run_name']}-{time_str}"

    wandb_logger = WandbLogger(
        project=config["global"]["project_name"],
        name=run_name,
        save_dir=config["global"]["save_dir"],
        entity=config["global"]["username"],
    )
    wandb_logger.watch((model))

    # only save on rank-0 process if run on multiple GPUs
    # https://github.com/Lightning-AI/lightning/issues/13166
    if rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(config)

    callbacks = [
        CALLBACK_REGISTRY.get(mcfg["name"])(**mcfg["args"])
        for mcfg in config["callbacks"]
    ]
    trainer = pl.Trainer(
        default_root_dir=".",
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"] 
            if config["trainer"]["accumulate_grad_batches"] else 1,
        max_epochs=config["trainer"]["num_epochs"],
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        strategy=config["trainer"]["strategy"],
        check_val_every_n_epoch=config["trainer"]["evaluate_interval"],
        log_every_n_steps=config["trainer"]["log_interval"],
        enable_checkpointing=True,
        precision=config["trainer"]["precision"],
        fast_dev_run=config["trainer"]["debug"],  # turn on if you only want to debug
        logger=wandb_logger,
        callbacks=callbacks,
    )
    
    trainer.fit(model, ckpt_path=config["global"]["resume"])

    return wandb_logger.experiment.id


if __name__ == "__main__":
    cfg = Opts(cfg="configs/train/config_cldrive_v1.yaml").parse_args()
    seed_everything(seed=cfg["global"]["SEED"])
    # setting location for temporary directory
    tempfile.tempdir = cfg["global"]["temp_dir"]
    train(cfg)
