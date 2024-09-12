import pandas as pd
from loguru import logger
import numpy as np
import tqdm

concatenated_df = pd.read_csv("full_cldrive.csv")
pass_instances = concatenated_df[(concatenated_df["outcome"] == "PASS")]
pass_instances.to_csv("pass_cldrive.csv", index=False)


# pass_instances = pd.read_csv('pass_cldrive.csv')
# Function to calculate the statistics for each kernel instances
def calculate_stats(row):
    try:
        times = np.array(eval(row["kernel_time_ns"]))
    except Exception as e:
        logger.error(f"Error in calculating stats for {row['kernel_path']}")
        logger.error(e)
        return None
    max_time = np.max(times)
    min_time = np.min(times)
    mean_time = np.mean(times)
    std = np.std(times)
    percentile_25 = np.percentile(times, 25)
    percentile_50 = np.percentile(times, 50)
    percentile_75 = np.percentile(times, 75)
    percentile_99 = np.percentile(times, 99)
    values = [
        max_time,
        min_time,
        mean_time,
        std,
        percentile_25,
        percentile_50,
        percentile_75,
        percentile_99,
    ]
    columns = [
        "max_time",
        "min_time",
        "mean_time",
        "std",
        "percentile_25",
        "percentile_50",
        "percentile_75",
        "percentile_99",
    ]
    for i, col in enumerate(columns):
        row[col] = values[i]
    return row


# Apply the function to each row and add the resulting columns to the DataFrame
print("Calculating statistics for each kernel instance...")
pass_instances = pass_instances.apply(calculate_stats, axis=1)
pass_instances = pass_instances.dropna()
pass_instances["norm_std"] = pass_instances["std"] / pass_instances["mean_time"]

# Print the resulting DataFrame with calculated statistics
print("Saving the resulting DataFrame...")
print(pass_instances.head(0))
pass_instances.to_csv("pass_stats_cldrive.csv", index=False)
