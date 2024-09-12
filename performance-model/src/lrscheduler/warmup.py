from transformers import get_linear_schedule_with_warmup


def Warmup(optimizer, config):
    lr_scheduler = {
        "scheduler": get_linear_schedule_with_warmup(
            optimizer,
            **config["args"],
            # num_warmup_steps=config["warmup_steps"],
        ),
        "name": "learning_rate_warmup",
        "interval": config["interval"],
        "frequency": config["frequency"],
    }

    return lr_scheduler
