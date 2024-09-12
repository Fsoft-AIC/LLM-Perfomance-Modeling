# LLMPerf: GPU Performance Modeling meets Large Language Models
This repository contains the source code for creating large-scale performance datasets, training and evaluating the LLMPerf model. The model is used to predict the performance of OpenCL kernels on a specific device.

The folder structure is as follows:
- `performance-model`: Source code for training and evaluating the LLMPerf model.
- `cldrive`: Source code for automatically running OpenCL kernels on a specific device and collecting performance data.
- `mem-access-analysis`: Source code for inserting memory access hook into OpenCL kernels, used to collect memory access traces and generate bound-aware datasets.

## Citation
If you use this codebase, or otherwise find our work valuable, please cite our paper:
```
@inproceedings{nguyen2024llmperf,
  title={LLMPerf: GPU Performance Modeling meets Large Language Models},
  author={Nguyen-Nhat, Minh-Khoi and Do, Hoang Duy Nguyen and Le, Huyen Thao and Dao, Thanh Tuan},
  booktitle={Proceedings of the International Symposium on the Modeling, Analysis, and Simulation of Computer and Telecommunication Systems},
  year={2024},
  organization={IEEE}
}
