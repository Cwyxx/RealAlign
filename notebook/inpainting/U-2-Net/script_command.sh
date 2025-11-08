#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=6

# prompt_file="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/qwen_3_caption-general_append/chameleon_real_qwen3_caption_results.csv"
# input_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/chameleon_real-random_add_noise_step/real"
# output_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/chameleon_real"
# python u2net_next_inpainting.py --input_dir ${input_dir} --output_dir ${output_dir} --prompt_file ${prompt_file}

# prompt_file="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/ava_dataset/ava_dataset_qwen3_caption_results.csv"
# input_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/aesthetic_visual_analysis/AVA_dataset/high_aesthetic_images"
# output_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/ava_dataset"
# python u2net_next_inpainting.py --input_dir ${input_dir} --output_dir ${output_dir} --prompt_file ${prompt_file}

num_inference_steps=10
prompt_file="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset/high_quality_train.csv"
input_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step/real"
output_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting_${num_inference_steps}/general_1"
python u2net_next_inpainting.py --input_dir ${input_dir} --output_dir ${output_dir} --prompt_file ${prompt_file} --num_inference_steps ${num_inference_steps}