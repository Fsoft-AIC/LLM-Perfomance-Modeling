from src.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")

from src.dataset.local_code_dataset import LocalCodeDataset
from src.dataset.hf_code_dataset import HFCodeDataset
