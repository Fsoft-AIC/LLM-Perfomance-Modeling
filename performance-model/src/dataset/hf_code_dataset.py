import pandas as pd
from datasets import load_dataset

from src.utils.preprocess import preprocess_data_opencl, remove_exceed_length
from . import DATASET_REGISTRY


@DATASET_REGISTRY.register()
def HFCodeDataset(config, tokenizer):
    """
    Generate code dataset from jsonl file.

    Args:
        config (_type_): Must contain these fields: `dataset`, `subset`, `split`, `max_seq_len`, `output_scale`
        tokenizer (_type_): tokenzier from transformers

    Returns:
        _type_: _description_
    """
    if config["data_source"] == "huggingface":
        dts = load_dataset(config["dataset"], config["subset"], split=config["split"])
    else:
        raise ValueError("Data type not supported")
    dts = dts.map(
        lambda x: preprocess_data_opencl(
            x,
            tokenizer,
            max_seq_len=config["max_seq_len"],
            output_scale=config["output_scale"],
        ),
        batched=True,
    ).map(
        lambda x: remove_exceed_length(x, config["max_seq_len"]), 
        batched=True
    )
    dts.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "execution_time"],
        output_all_columns=True,
    )
    return dts
