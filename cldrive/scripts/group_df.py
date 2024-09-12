import json
import pandas as pd
import os
from tqdm import tqdm
from loguru import logger
import numpy as np

# Define the folder path where your CSV files are located
result_dir = "./local/backup-github-1M1/success"
kernel_dir = "./local/kernels"

def get_data(filename):
    if not filename.endswith(".csv"): return None
    file_path = os.path.join(result_dir, filename)
    df = pd.read_csv(file_path)
    # group by device,kernel,global_size,local_size_x,args_info
    aggr = {}
    aggr['kernel_time_ns'] = 'mean'
    aggr['outcome'] = lambda x: x.iloc[0]
    df = df.groupby(['device','kernel','global_size','local_size_x','args_info'])\
            .agg(aggr).reset_index()
    kernel_id = os.path.splitext(filename)[0].split("_")[0]
    with open(os.path.join(kernel_dir, f"{kernel_id}.cl"), "r", encoding="utf-8") as f:
        df["kernel_code"] = f.read()
    return df

# multiprocess
from multiprocessing import Pool
import multiprocessing

files = list(os.listdir(result_dir))[:10]
df_list = []
with Pool(multiprocessing.cpu_count() // 4) as p:
    for df in tqdm(p.imap(get_data, files), total=len(files)):
        if df is not None:
            df_list.append(df)

# Concatenate the individual DataFrames into one DataFrame
concatenated_df = pd.concat(df_list, ignore_index=True)

# Print the concatenated DataFrame
print(len(concatenated_df))

concatenated_df.to_csv('test_cldrive.csv', index=False)

outcome_counts = concatenated_df['outcome'].value_counts()

for category, count in outcome_counts.items():
    logger.info(f'{category}: {count}')
