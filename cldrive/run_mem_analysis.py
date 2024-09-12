import io
import json
import sys
import pandas as pd
import os
from tqdm import tqdm
import multiprocessing
import random
from loguru import logger
import numpy as np
import copy
import itertools

from app.parser import ParseCLDriveStdoutToDataframe
from app.utils import detect_kernel_dimensions, getOpenCLPlatforms, get_kernel_info
from app.runner import RunCLDrive

CLDRIVE = "bazel-bin/gpu/cldrive/cldrive"
CLMEM = "bazel-bin/gpu/clmem/clmem"
ORG_KERNEL_DIR = "kernels"
HOOK_INSERTED_KERNEL_DIR = "kernels-modified"
BACKUP_DIR = "backup-mem-analysis"
SUCCESS_DIR = os.path.join(BACKUP_DIR, "success")
FAIL_DIR = os.path.join(BACKUP_DIR, "fail")
TIMEOUT = 10
NUM_GPU = 1
NUM_PROCESS = 8
verbose_cldrive = False

random.seed(2610)

# logger.remove(0)
logger.add("cldrive.log", level="INFO")
logger.add(sys.stderr, level="DEBUG")
rng = np.random.default_rng()

def get_config():
    local_sizes = [32]
    global_sizes = [192, 512]

    def gen_launch_configs():
        launch_configs = []
        for lsize in local_sizes:
            for gsize in global_sizes:
                launch_configs.append((gsize, lsize))
        return launch_configs
    
    return gen_launch_configs()

class KernelRunInstance:
    def __init__(self, kernel_code, gsize, lsize, args_values=None) -> None:
        self.kernel_code = kernel_code
        self.gsize = gsize
        self.lsize = lsize
        self.args_info = args_values
    
    def run_mem_access(self):
        stdout, stderr = RunCLDrive(
            cldrive_exe=CLDRIVE,
            src=self.kernel_code,
            num_runs=1,
            lsize=self.lsize,
            gsize=self.gsize,
            args_values=self.args_info,
            cl_platform=getOpenCLPlatforms(CLDRIVE)[0],
            timeout=TIMEOUT,
            output_format="null",
        )

        return stdout, stderr
    
    def run_check(self):
        stdout, stderr = RunCLDrive(
            cldrive_exe=CLDRIVE,
            src=self.kernel_code,
            num_runs=1,
            lsize=self.lsize,
            gsize=self.gsize,
            args_values=self.args_info,
            cl_platform=getOpenCLPlatforms(CLDRIVE)[0],
            timeout=TIMEOUT,
            output_format="csv",
        )
        df, stderr = ParseCLDriveStdoutToDataframe(stdout, stderr)

        return df, stderr
    

class KernelMemoryAnalyzer:
    TEST_GLOBAL_ARRAY_BOUND = 1e6
    TEST_LOCAL_ARRAY_BOUND = 512
    def __init__(self, kernel_path) -> None:
        self.kernel_path = kernel_path
        with open(kernel_path, "r", encoding="utf-8") as f:
            self.kernel_code = f.read()
        self.argument_list = self.get_argument_list()
        self.id2name = {arg["id"]: arg["name"] for arg in self.argument_list}
        self.scalar_argument_list = [arg["id"] for arg in self.argument_list if arg["is_pointer"] == False]
    
    def get_argument_list(self):
        # tuple of (kernel_name, argument_desc)
        # argument_desc is a dictionary of keys 'id', 'name', 'type', 'qualifier', 'is_pointer'
        kernels = list(get_kernel_info(CLDRIVE, self.kernel_path)[self.kernel_path].items())
        # get the first kernel argument desc, currently only support one kernel per file
        return kernels[0][1]
    
    def get_run_settings(self, gsize, lsize):
        """
        1. List candidate value for scalar arguments
        scalar usually follow the memory access pattern of the kernel:
          1. array bound => gsize
          2. offset + stride => [1, 2, 4,...]
          3. number of elements per one work-item => [1, 2, 4,...]
        2. For each combination of scalar values, run the kernel and get the memory access pattern
        3. Fit the memory access pattern into a linear function
        4. Return a set of combinations that have array bound larger or equal to gsize
        """  
        candidate_values = [1, 4, gsize, 16, 32, 256]
        
        arg_settings = []
        n_tried = 0
        for scalars_setting in itertools.product(candidate_values, repeat=len(self.scalar_argument_list)):
            # get the linear equation of array bound
            argument_list = self.get_array_bound_relation(scalars_setting)
            
            if len(argument_list) == 0:
                continue # skip if no array is used or run failed
            
            # create argument value for cldrive + check if array bound is larger or equal to gsize
            args_values = []
            i = 0
            is_geq_than_gsize = True
            
            for argument in argument_list:
                if argument["is_pointer"] == False:
                    args_values.append(scalars_setting[i])
                    i += 1
                else:
                    args_values.append(int(np.ceil(argument["coef"][0] + argument["coef"][1]*gsize)))
                    # if args_values[-1] < gsize:
                    if args_values[-1] < 0:
                        is_geq_than_gsize = False
                        break
            
            # if array bound is larger or equal to gsize, add to the list
            if is_geq_than_gsize:
                arg_settings.append(args_values)
            n_tried += 1
            if n_tried > 1000: break
                
        return arg_settings
    
    def get_array_bound_relation(self, scalars_setting):              
        try:
            X, ys = self.get_mem_access_dataset(scalars_setting)
        
            result = copy.deepcopy(self.argument_list)
            
            for i in range(len(result)):
                arg_info = result[i]
                # skip scalar arguments
                if arg_info["is_pointer"] == False: continue
                # if array is not used, use gsize as the bound
                if arg_info["name"] not in ys.keys(): 
                    result[i]["coef"] = [0, 1]
                else:
                    arg_name = arg_info["name"]
                    y = ys[arg_name]
                    # fit datapoint into a linear function
                    # array_bound = a*gsize + b
                    coef = np.linalg.solve(X, y)
                    result[i]["coef"] = coef.tolist()
                
            return result
        except Exception as e:
            logger.error(f"ERROR: (`{self.kernel_path}`, gsize={gsize}, lsize={lsize}) Get memory access dataset failed with error:\n{e}")
            return {}
    
    def get_mem_access_dataset(self, scalars_setting):
        launch_configs = list(get_config())
        max_ids, min_ids, launch_configs = self.run_multiple_configs(launch_configs, scalars_setting)
        X = np.array(launch_configs)[:,0] # only use gsize
        X = np.expand_dims(X, axis=1) # expand to (2,1)
        X = np.insert(X, 0, 1, axis=1) # add coef column = 1
        ys = pd.DataFrame(max_ids).to_dict("list")
        return X, ys
    
    def run_multiple_configs(self, launch_configs, scalars_setting):
        run_results = []
        for launch_config in launch_configs:
            run_results.append(self.get_max_min_of_run(launch_config, scalars_setting))
        max_list, min_list, launch_config = zip(*run_results)

        return max_list, min_list, launch_config
            
    def get_max_min_of_run(self, launch_config, scalars_setting):
        gsize, lsize = launch_config
        args_values = []
        
        i = 0
        for arg_info in self.argument_list:
            if arg_info["is_pointer"] == False:
                args_values.append(int(scalars_setting[i]))
                i += 1
            else:
                if arg_info["qualifier"].strip("_") == "global" or arg_info["qualifier"].strip("_") == "constant":
                    args_values.append(int(KernelMemoryAnalyzer.TEST_GLOBAL_ARRAY_BOUND))
                else:
                    args_values.append(int(KernelMemoryAnalyzer.TEST_LOCAL_ARRAY_BOUND))
        kernel_instance = KernelRunInstance(self.kernel_code, gsize, lsize, args_values)
        stdout, stderr = kernel_instance.run_mem_access()
        if len(stdout) == 0:
            raise Exception(f"Run kernel with args_values={args_values} failed with error:\n{stderr}")
        # for each memory access pair (arg_id, id_access), get the max and min of the run
        # return a list of (arg_id, max, min)
        max_list, min_list = self.parse_stdout(stdout)
        max_dict = {self.id2name[x["arg_id"]]: x["id_access"] for x in max_list}
        min_dict = {self.id2name[x["arg_id"]]: x["id_access"] for x in min_list}
        return max_dict, min_dict, launch_config
        
    def parse_stdout(self, stdout):
        # parse the stdout of CLDrive
        # return a list of (arg_id, id_access)
        stdout = "arg_id,id_access\n" + stdout
        df = pd.read_csv(io.StringIO(stdout))
        df_group = df.groupby("arg_id")
        max_list = df_group.agg({"id_access":'max'}).reset_index().to_dict("records")
        min_list = df_group.agg({"id_access":'min'}).reset_index().to_dict("records")
        return max_list, min_list


def set_cuda_visible():
    process_number = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(process_number)

def get_kernel_args_values(args):
    kernel, gsize, lsize = args
    hook_inserted_kernel_path = os.path.join(HOOK_INSERTED_KERNEL_DIR, kernel)
    org_kernel_path = os.path.join(ORG_KERNEL_DIR, kernel)
    logger.info(f"Analyzing {org_kernel_path}")
    
    try:
        analyzer = KernelMemoryAnalyzer(hook_inserted_kernel_path)
        
        run_settings = analyzer.get_run_settings(gsize, lsize)
        save_dir = SUCCESS_DIR if len(run_settings) > 0 else FAIL_DIR
        with open(os.path.join(save_dir, os.path.splitext(kernel)[0] + ".json"), "w", encoding="utf-8") as f:
            json.dump(run_settings, f)
    except Exception as e:
        logger.error(f"ERROR: Analyze kernel `{org_kernel_path}` failed with error:\n{e}")
    
    # for run_setting in run_settings:
    #     with open(org_kernel_path, "r", encoding="utf-8") as f:
    #         kernel_code = f.read()
    #     kernel_instance = KernelRunInstance(kernel_code, gsize, lsize, run_setting)
    #     df, stderr = kernel_instance.run_check()
    # res_path = os.path.join(BACKUP_DIR, os.path.splitext(kernel)[0] + ".json")
    # with open(res_path, "w", encoding="utf-8") as f:
    #     json.dump(res_dict, f)

if __name__ == "__main__":
    BACKUPED_LIST = set()
    if not os.path.exists(SUCCESS_DIR):
        os.makedirs(SUCCESS_DIR, exist_ok=True)
    if not os.path.exists(FAIL_DIR):
        os.makedirs(FAIL_DIR, exist_ok=True)
    else:
        BACKUPED_LIST = set(os.path.splitext(kernel)[0] for kernel in os.listdir(SUCCESS_DIR))
        BACKUPED_LIST.update(os.path.splitext(kernel)[0] for kernel in os.listdir(FAIL_DIR))
    with open("selected_kernels-200k.json", "r") as f:
        SELECTED_KERNEL = set(os.path.splitext(os.path.basename(kernel))[0] for kernel in json.load(f))
    
    need_calculate = []
    for kernel in os.listdir(HOOK_INSERTED_KERNEL_DIR):
        kernel_path = os.path.join(HOOK_INSERTED_KERNEL_DIR, kernel)
        with open(kernel_path, "r", encoding="utf-8") as f:
            kernel_code = f.read()
        if os.path.splitext(kernel_path)[1] != ".cl" \
            or os.path.splitext(kernel)[0] in BACKUPED_LIST \
            or detect_kernel_dimensions(kernel_code) != "1D" \
            or os.path.splitext(kernel)[0] not in SELECTED_KERNEL:
            continue
        need_calculate.append(kernel)
    
    lsize = 32 * 3
    gsize = lsize * 13

    with multiprocessing.Pool(processes=NUM_PROCESS, initializer=set_cuda_visible) as pool:
        # Use the pool to map the process_element function to the elements
        
        pool.map(
            get_kernel_args_values, 
            zip(need_calculate, itertools.repeat(gsize), itertools.repeat(lsize)),
            # chunksize=NUM_PROCESS
        )
            
