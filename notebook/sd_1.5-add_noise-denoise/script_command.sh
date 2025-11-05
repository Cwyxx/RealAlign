#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=6

prompt_file="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/konIQ-10k/koniq_10k_qwen3_caption_results.csv"
input_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/konIQ-10k/1024x768"
output_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/konIQ-10k"
python add_noise-denoise.py --input_dir ${input_dir} --output_dir ${output_dir} --prompt_file ${prompt_file}
