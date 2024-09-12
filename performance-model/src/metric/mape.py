import torch
from src.metric import METRIC_REGISTRY


from torchmetrics import (
    MeanAbsolutePercentageError as MeanAbsolutePercentageErrorMetric,
)


@METRIC_REGISTRY.register()
class MeanAbsolutePercentageError(MeanAbsolutePercentageErrorMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MAPE"


if __name__ == "__main__":
    x = MeanAbsolutePercentageError()
    print(x.name)
