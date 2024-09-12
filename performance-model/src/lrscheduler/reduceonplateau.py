from torch.optim import lr_scheduler


def ReduceOnPlateau(optimizer, config):
    scheduler = {
        "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, **config["args"]),
        "monitor": config["monitor"],
        "interval": config["interval"],
    }

    return scheduler
