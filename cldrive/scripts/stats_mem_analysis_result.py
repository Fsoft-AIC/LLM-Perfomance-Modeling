import itertools
import json
import os

import numpy as np

from app.utils import get_kernel_info

CLDRIVE = "bazel-bin/gpu/cldrive/cldrive"
BACKUP_DIR = "backup-mem-analysis"
SUCCESS_DIR = os.path.join(BACKUP_DIR, "success")
FAIL_DIR = os.path.join(BACKUP_DIR, "fail")

def get_argument_list(kernel_path):
  # tuple of (kernel_name, argument_desc)
  # argument_desc is a dictionary of keys 'id', 'name', 'type', 'qualifier', 'is_pointer'
  kernels = list(get_kernel_info(CLDRIVE, kernel_path)[kernel_path].items())
  # get the first kernel argument desc, currently only support one kernel per file
  return kernels[0][1]

with open("selected_kernels-200k.json", "r") as f:
  SELECTED_KERNEL = set(os.path.splitext(os.path.basename(kernel))[0] for kernel in json.load(f))

if __name__ == "__main__":
  success_kernels = list(os.path.splitext(os.path.basename(kernel))[0] 
                         for kernel in os.listdir(SUCCESS_DIR)
                         if os.path.splitext(os.path.basename(kernel))[0] in SELECTED_KERNEL)
  fail_kernels = list(os.path.splitext(os.path.basename(kernel))[0] 
                         for kernel in os.listdir(FAIL_DIR)
                         if os.path.splitext(os.path.basename(kernel))[0] in SELECTED_KERNEL)
  print("Success kernels: {}".format(len(success_kernels)))
  print("Fail kernels: {}".format(len(fail_kernels)))
  
  result_stats = []
  gsize = 1248
  for kernel in success_kernels:
    path = os.path.join(SUCCESS_DIR, kernel + ".json")
    argument_list = get_argument_list(os.path.join("kernels-modified", kernel + ".cl"))
    
    with open(path, "r") as f:
      mem_analysis_result = json.load(f)
      # count the number of settings by count the number of scalar (non-pointer) arguments
      scalar_argument_list = [arg for arg in argument_list if not arg["is_pointer"]]
      candidate_values = [1, 4, gsize, 16, 32, 256]
      
      n_settings = 6**len(scalar_argument_list) if len(scalar_argument_list) <= 3 else 1001
      
      result_stats.append((len(mem_analysis_result), n_settings))
  
  # each is tuple of (n_success, n_settings)
  results_stats = np.array(result_stats)
  # percentage of success for each kernel
  success_rate = results_stats[:,0] / results_stats[:,1]
  # masking for kernels with n_settings > 1000
  mask = results_stats[:,1] > 1000
  # plot the success rate bar plot for kernels, x-axis is the kernel index
  # y-axis is the success rate, kernel with mask = 1 have red color, 
  # kernel with mask = 0 have blue color 
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(20, 10))
  plt.bar(np.arange(len(success_kernels)), success_rate, color=["blue" if m == 0 else "red" for m in mask])
  plt.ylabel("Success rate")
  plt.savefig("success_rate.png")
