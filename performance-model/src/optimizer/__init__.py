from src.utils.registry import Registry

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")

from src.optimizer.adamw import TorchAdamW

OPTIMIZER_REGISTRY.register(TorchAdamW)
