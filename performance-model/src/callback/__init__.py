from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    ModelSummary,
)
from src.utils.registry import Registry

CALLBACK_REGISTRY = Registry("CALLBACK")

# https://lightning.ai/docs/pytorch/2.0.0/api/lightning.pytorch.callbacks.EarlyStopping.html
CALLBACK_REGISTRY.register(EarlyStopping)
# https://lightning.ai/docs/pytorch/2.0.0/api/lightning.pytorch.callbacks.ModelCheckpoint.html
CALLBACK_REGISTRY.register(ModelCheckpoint)
# https://lightning.ai/docs/pytorch/2.0.0/api/lightning.pytorch.callbacks.LearningRateMonitor.html
CALLBACK_REGISTRY.register(LearningRateMonitor)
# https://lightning.ai/docs/pytorch/2.0.0/api/lightning.pytorch.callbacks.ModelSummary.html
CALLBACK_REGISTRY.register(ModelSummary)
