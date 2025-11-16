#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=7

accelerate launch --num_processes 1 -m tools.pick_a_pic --score pickscore --output /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO
