import pandas as pd
from datasets import Dataset

from src.utils.preprocess import preprocess_data_opencl, remove_exceed_length
from . import DATASET_REGISTRY


@DATASET_REGISTRY.register()
def LocalCodeDataset(config, tokenizer):
    """
    Generate code dataset from jsonl file.

    Args:
        config (_type_): Must contain these fields: `data_path`, `max_seq_len`, `output_scale`
        tokenizer (_type_): tokenzier from transformers

    Returns:
        _type_: _description_
    """
    df = pd.read_json(config["data_path"], lines=True, orient="records")
    dts = Dataset.from_pandas(df).map(
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
