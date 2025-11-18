#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=5,6

accelerate launch --multi_gpu --num_processes 2 train.py --train_dataset \
    --logdir