#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=6

prompt_file="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset/high_quality_train.csv"
input_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset/high_quality_train/real"
output_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise-random_denoise"
python random_add_noise-random_denoise.py --input_dir ${input_dir} --output_dir ${output_dir} --prompt_file ${prompt_file}
