#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 

# 单 GPU 启动
export CUDA_VISIBLE_DEVICES=7
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train-sd-3-5-medium-irl.py --config config/sd3_5_medium_irl.py:paired_real_fake_dataset_sd3

# 多 GPU 启动（4卡）
# export CUDA_VISIBLE_DEVICES=1,2,3,4
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train-sd-3-5-medium-irl.py --config config/sd3_5_medium_irl.py:paired_real_fake_dataset_sd3