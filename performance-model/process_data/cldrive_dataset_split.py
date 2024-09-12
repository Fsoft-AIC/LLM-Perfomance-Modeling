import pandas as pd
import random
import argparse
import os
import json

if __name__ == "__main__":
    # Define parser arguments
    parser = argparse.ArgumentParser(
        description="Split and preprocess OpenCL kernel data."
    )
    parser.add_argument(
        "--train_percentage",
        type=float,
        default=0.9,
        help="Percentage of training data",
    )
    parser.add_argument(
        "--source_file",
        type=str,
        help="Source CSV file",
    )
    parser.add_argument(
        "--target_directory",
        type=str,
        help="Target directory for JSONL files",
    )

    args = parser.parse_args()

    # Set the TRAIN_PERCENTAGE based on the command-line argument
    TRAIN_PERCENTAGE = args.train_percentage

    # Load your CSV data into a pandas DataFrame
    df = pd.read_csv(args.source_file)
    # filter out the rows with outcome not equals "PASS"
    df = df[df["outcome"] == "PASS"]
    df["code"] = df["kernel_code"]
    df["execution_time"] = df["kernel_time_ns"]
    df["gsize"] = df["global_size"]
    df["lsize"] = df["local_size_x"]

    # Get the unique kernel_path values
    unique_kernel_paths = df["kernel_code"].unique()

    # Shuffle the unique kernel_path values randomly
    random.shuffle(unique_kernel_paths)

    # Determine the split point
    split_point = int(len(unique_kernel_paths) * TRAIN_PERCENTAGE)

    # Split the shuffled kernel_path values into train and validation sets
    train_kernel_paths = unique_kernel_paths[:split_point]
    validation_kernel_paths = unique_kernel_paths[split_point:]

    # Create the training and validation sets
    train_set = df[df["kernel_code"].isin(train_kernel_paths)]
    val_set = df[df["kernel_code"].isin(validation_kernel_paths)]

    def create_input_desc(row):
        args_info = json.loads(row["args_info"])
        args_desc = []
        for arg_info in args_info:
            desc = ""
            desc += f"Argument at position {arg_info['id']} is `{arg_info['name']}`, which is {arg_info['qualifier']} "
            if arg_info['is_pointer']:
                desc += f"buffer of type `{arg_info['type']}` with size `{arg_info['value']}`"
            else:
                desc += f"scalar of type `{arg_info['type']}` with value `{arg_info['value']}`"
            args_desc.append(desc)
        return "\n".join(args_desc)

    train_set["input_sizes"] = train_set.apply(create_input_desc, axis=1)
    val_set["input_sizes"] = val_set.apply(create_input_desc, axis=1)

    # Save the sets to JSONL format in the specified target directory
    os.makedirs(args.target_directory, exist_ok=True)
    train_set.to_json(
        os.path.join(args.target_directory, "opencl_train.jsonl"),
        orient="records",
        lines=True,
    )
    val_set.to_json(
        os.path.join(args.target_directory, "opencl_val.jsonl"),
        orient="records",
        lines=True,
    )
