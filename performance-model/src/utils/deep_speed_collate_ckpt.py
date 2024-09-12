from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import argparse

default_save_path = "local/performance-modeling/u15dsudb/checkpoints/codegen350-epoch=15-val_mape=0.1006-val_loss=0.0329.ckpt"
default_output_path = "local/performance-modeling/u15dsudb/checkpoints/codegen350-epoch=15-val_mape=0.1006-val_loss=0.0329.ckpt/checkpoint.pt"

# lightning deepspeed has saved a directory instead of a file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source_dir",
        type=str,
        default=default_save_path,
        help="path to the desired checkpoint folder, e.g., path/checkpoint-12",
    )
    parser.add_argument(
        "-t",
        "--target_file",
        type=str,
        default=default_output_path,
        help="path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)",
    )
    args = parser.parse_args()
    convert_zero_checkpoint_to_fp32_state_dict(args.source_dir, args.target_file)
