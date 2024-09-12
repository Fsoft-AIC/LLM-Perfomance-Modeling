from torch.optim import AdamW


def TorchAdamW(model_parameters, **adam_args):
    optimizer = AdamW(model_parameters, **adam_args)

    return optimizer
