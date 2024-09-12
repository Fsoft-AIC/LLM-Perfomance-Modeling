import json
import random
import os
import Levenshtein
from torch import unique
from tqdm import tqdm

KERNEL_DIR = "../kernels"
device_num_sm = 82 # Nvidia V100
# device_num_sm = 108 # Nvidia A100

def remove_duplicate_kernels(kernels):
    """
    Remove duplicate kernels from a dict of {kernel_path: kernel_code}
    """
    unique_kernels = {}
    for kernel in tqdm(kernels):
        is_duplicate = False
        kernel_code = kernels[kernel]
        for unique_kernel in unique_kernels:
            unique_kernel_code = unique_kernels[unique_kernel]
            if Levenshtein.ratio(kernel_code, unique_kernel_code, score_cutoff=0.95) != 0:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_kernels[kernel] = kernel_code
    return unique_kernels


def detect_kernel_dimensions(kernel_code):
    dimensions = {
        "3D-global": ["get_global_id(2)", "get_global_size(2)", 
                      "get_group_id(2)", "get_num_groups(2)", "get_global_offset(2)"],
        "2D-global": ["get_global_id(1)", "get_global_size(1)", 
                      "get_group_id(1)", "get_num_groups(1)", "get_global_offset(1)"],
        "3D-local": ["get_local_id(2)", "get_local_size(2)"],
        "2D-local": ["get_local_id(1)", "get_local_size(1)"],
    }

    result = ""
    for dim, patterns in dimensions.items():
        if any(pattern in kernel_code for pattern in patterns):
            result += " " + dim
    if result == "":
        result = "1D"

    return result.strip()

def remove_multi_dimension_kernels(kernels):
    """
    Remove multi-dimensional kernels from a dict of {kernel_path: kernel_code}
    """
    unique_kernels = {}
    for kernel in tqdm(kernels):
        kernel_code = kernels[kernel]
        if detect_kernel_dimensions(kernel_code) == "1D":
            unique_kernels[kernel] = kernel_code
    return unique_kernels

def get_config():
    # including both simple case (multiple of 32) and complex case (not multiple of 32)
    warp_size = 32
    n_sample_local = 4
    n_sample_wg = 50
    local_sizes = range(warp_size * 1, warp_size * 32, warp_size)
    wg_sizes = range(1, device_num_sm*50)
    MAX_GSIZE = int(1e7) - 1

    def gen_launch_configs():
        launch_configs = []
        l_samples = random.sample(local_sizes, n_sample_local)
        wg_samples = random.sample(wg_sizes, n_sample_wg)

        for lsize in l_samples:
            for wg_size in wg_samples:
                gsize = lsize * wg_size
                if gsize > MAX_GSIZE:
                    gsize = lsize * int(MAX_GSIZE / lsize)
                launch_configs.append((gsize, lsize))
        return launch_configs

    def gen_kernels():
        # Get a list of all files and subdirectories in the specified directory
        all_kernels = list(os.listdir(KERNEL_DIR))
        print(f"Number of kernels: {len(all_kernels)}")
        # build kernel_path - kernel_code dict
        kernels = {}
        for kernel_path in all_kernels:
            with open(os.path.join(KERNEL_DIR, kernel_path)) as f:
                kernels[kernel_path] = f.read()
        # choose only 1D kernel
        print(f'Removing multi-dimensional kernels...')
        selected_kernels = remove_multi_dimension_kernels(kernels)
        print(f'Removed multi-dimensional kernels. Number of kernels: {len(selected_kernels)}')
        # remove duplicate kernels
        print(f'Removing duplicate kernels...')
        selected_kernels = remove_duplicate_kernels(selected_kernels)
        print(f'Removed duplicate kernels. Number of kernels: {len(selected_kernels)}')
        return selected_kernels
        
    kernel_path_configs = gen_kernels()
    for kernel in kernel_path_configs:
        launch_configs = gen_launch_configs()
        for launch_config in launch_configs:
            yield [
                kernel,
                launch_config[0],
                launch_config[1],
            ]

if __name__ == "__main__":
    configs = list(get_config())
    print('Number of configs:', len(configs))
    
    with open("default_configs.json", "w") as f:
        json.dump(list(configs), f)
