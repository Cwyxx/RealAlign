#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=0

prompt_file="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/qwen_3_caption-general_append/chameleon_real_qwen3_caption_results.csv"
input_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/chameleon_real-random_add_noise_step/real"
output_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/chameleon_real"
python u2net_next_inpainting.py --input_dir ${input_dir} --output_dir ${output_dir} --prompt_file ${prompt_file}
