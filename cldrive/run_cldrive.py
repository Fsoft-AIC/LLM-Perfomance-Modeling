import json
import os
from tqdm import tqdm
import random
from loguru import logger
import numpy as np
import multiprocessing

from app.runner import KernelRunInstance

KERNEL_DIR = 'local/kernels'
CONFIG_PATH = "local/default_configs.json"
BACKUP_DIR = "local/backup-github-1M1"
SUCCESS_DIR = os.path.join(BACKUP_DIR, "success")
FAIL_DIR = os.path.join(BACKUP_DIR, "fail")
BACKUPED_LIST = set()

TIMEOUT = 60 # 1 minutes
N_RUN = 5

GPU_POOL = [1, 2, 3]
NUM_GPU = len(GPU_POOL)

random.seed(2610)

# logger.remove(0)
rng = np.random.default_rng()

def run_kernel(kernel_configs):
    kernel_path, lsize, gsizes = kernel_configs
    
    kernel_id = os.path.splitext(kernel_path)[0]
    
    kernel_code = open(os.path.join(KERNEL_DIR, kernel_path), "r", encoding="utf-8").read()
    
    for i, gsize in enumerate(gsizes):
        if f"{kernel_id}_{gsize}_{lsize}" in BACKUPED_LIST:
            continue

        run_instance = KernelRunInstance(kernel_code=kernel_code,
                                            gsize=gsize,
                                            lsize=lsize,
                                            timeout=TIMEOUT)
        result_df, stderr = run_instance.run_n_times(N_RUN)
        if result_df is None:
            with open(os.path.join(FAIL_DIR, f"{kernel_id}_{gsize}_{lsize}.txt"), "w") as f:
                f.write(stderr)
            # if stderr is TIMEOUT, we mark all remaining gsize of this kernel as TIMEOUT
            if "TIMEOUT" in stderr:
                for j in range(i+1, len(gsizes)):
                    with open(os.path.join(FAIL_DIR, f"{kernel_id}_{gsizes[j]}_{lsize}.txt"), "w") as f:
                        f.write("TIMEOUT")
                break
        else:
            result_df.to_csv(os.path.join(SUCCESS_DIR, f"{kernel_id}_{gsize}_{lsize}.csv"), index=None)

def set_cuda_visible():
    process_number = GPU_POOL[multiprocessing.current_process()._identity[0] - 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(process_number)

if __name__ == "__main__":
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    # list of [(kernel_path, gsize, lsize)]
    configs = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
    
    if not os.path.exists(SUCCESS_DIR):
        os.makedirs(SUCCESS_DIR, exist_ok=True)
    if not os.path.exists(FAIL_DIR):
        os.makedirs(FAIL_DIR, exist_ok=True)
    BACKUPED_LIST = set(os.path.splitext(os.path.basename(kernel))[0] for kernel in os.listdir(SUCCESS_DIR))
    BACKUPED_LIST.update(os.path.splitext(os.path.basename(kernel))[0] for kernel in os.listdir(FAIL_DIR))
    # remove already run configs
    print(f"Total {len(configs)} kernels in config file")
    need_to_run_configs = []
    for kernel_configs in configs:
        kernel_path, gsize, lsize = kernel_configs
        kernel_id = os.path.splitext(kernel_path)[0]
        if f"{kernel_id}_{gsize}_{lsize}" not in BACKUPED_LIST:
            need_to_run_configs.append((kernel_path, gsize, lsize))
    print(f"Total {len(need_to_run_configs)} kernels need to run")
                
    need_to_run_configs = sorted(need_to_run_configs, key=lambda x: (x[0], x[2], x[1]))
    # create list which group by (kernel_path, lsize)
    # result list [(kernel_path, lsize, [gsize1, gsize2, ...])]
    need_to_run_grouped_configs = []
    i = 0
    while (i < len(need_to_run_configs)):
        kernel_path, gsize, lsize = need_to_run_configs[i]
        group = [gsize]
        i += 1
        while (i < len(need_to_run_configs) and kernel_path == need_to_run_configs[i][0] and lsize == need_to_run_configs[i][2]):
            group.append(need_to_run_configs[i][1])
            i += 1
        need_to_run_grouped_configs.append((kernel_path, lsize, group))
    
    # run in parallel
    with multiprocessing.Pool(
        processes=NUM_GPU, initializer=set_cuda_visible
    ) as pool:
        # Use the pool to map the process_element function to the elements
        list(tqdm(pool.imap(run_kernel, need_to_run_grouped_configs), total=len(need_to_run_grouped_configs)))

    
            
