import pandas as pd
import os
from tqdm import tqdm
from loguru import logger
import numpy as np

# Define the folder path where your CSV files are located
folder_path = "./cldrive-backup"

# Create an empty list to store the individual DataFrames
df_list = []

# Loop through the files in the folder and read them into DataFrames
for filename in tqdm(list(os.listdir(folder_path))):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df_list.append(df)

# Concatenate the individual DataFrames into one DataFrame
concatenated_df = pd.concat(df_list, ignore_index=True)

# Print the concatenated DataFrame
print(len(concatenated_df))

logger.add("cldrive_statistic.log")

logger.info(concatenated_df.columns)

concatenated_df.to_csv('full_cldrive.csv', index=False)

logger.info("----EXECUTION STATUS STATISTICS----")
outcome_counts = concatenated_df['outcome'].value_counts()

for category, count in outcome_counts.items():
    logger.info(f'{category}: {count}')
