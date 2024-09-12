# LLMPerf: OpenCL Performance Modeling
This folder contains the source code for training and evaluating the LLMPerf model. The model is used to predict the performance of OpenCL kernels on a specific device.

## Environment setup
Install torch and other required libraries by executing the following command:
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install the source library by execute this command at this folder:
```bash
pip install .
```

For wandb logging, you need to login to wandb by executing the following command:
```bash
wandb login
```
or disabled wandb logging by putting environment variable `WANDB_MODE=disabled` before the training or evaluation command.

### Dataset
The dataset is at [Hugging Face](https://huggingface.co/datasets/minhkhoi1026/opencl-llmperf). The dataset includes sub-datasets for different experiments. For example, `github-200k` is the dataset for the experiment on the GitHub dataset with 200k samples. More details about the dataset can be found in the README of the dataset. To use the dataset, setting the true `dataset`, `subset`, and `split` in `dataset` section of the configuration file.

## Evaluation
Pretrained models are available at [Hugging Face](https://huggingface.co/minhkhoi1026/LLMPerf). You need to download the model and set the path to the model in the configuration file.

You can change the configuration of the evaluation process in `/configs/eval/2B_log2.yaml`. You can also create your own configuration file. After that, evaluate the model by executing the following command:
```bash
python src/evaluate.py -c <path_to_your_config>
```

## Training
You can change the configuration of the model and the training process in `/configs/train/2B_log2_huggingface.yaml`  for Hugging Face dataset and `/configs/train/2B_log2_github.yaml` for local dataset (`jsonl` file). You can also create your own configuration file. After that, train the model by executing the following command:
```bash
python src/train.py -c <path_to_your_config>
```
