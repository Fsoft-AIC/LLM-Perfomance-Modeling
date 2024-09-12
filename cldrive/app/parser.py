import csv
import io
import os

from loguru import logger
import pandas as pd

def ParseCLDriveStdoutToDataframe(
    stdout: str
) -> pd.DataFrame:
    """
    Parse CLDrive result and return pandas dataframe.
    """

    try:
        df = pd.read_csv(io.StringIO(stdout))
    except Exception as e:
        df = None

    return df

def get_file_id_from_config(config):
    kernel_id = os.path.splitext(os.path.basename(config["kernel_path"]))[0]
    return f"{kernel_id}_{config['gsize']}_{config['lsize']}"
