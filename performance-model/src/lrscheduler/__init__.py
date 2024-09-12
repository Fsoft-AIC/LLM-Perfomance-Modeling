from src.utils.registry import Registry

LRSCHEDULER_REGISTRY = Registry("LRSCHEDULER")

from src.lrscheduler.reduceonplateau import ReduceOnPlateau
from src.lrscheduler.warmup import Warmup

LRSCHEDULER_REGISTRY.register(ReduceOnPlateau)
LRSCHEDULER_REGISTRY.register(Warmup)
