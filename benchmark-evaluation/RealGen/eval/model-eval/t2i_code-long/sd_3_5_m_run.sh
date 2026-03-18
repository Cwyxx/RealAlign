#!/bin/bash

# SD-3.5-M RealBench Long Prompt Generation Script
# Usage: bash sd_3_5_m_run.sh [cuda_device] [method] [checkpoint] [rl_framework]

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

cuda_device=${1:-0}
method=${2:-"sd-3-5-medium"}
ckpt=${3:-0}
rl_framework=${4:-"diffusion-dpo"}

export CUDA_VISIBLE_DEVICES=${cuda_device}
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

# Paths configuration
prompt_file="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/benchmark-evaluation/RealGen/eval/prompt.txt"
base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/model-ckpt"
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/generate_images/realgen"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"

# Generate images
python sd_3_5_m_t2i.py \
    --checkpoint_path "${ckpt_dir}" \
    --output_dir "${image_dir}" \
    --prompt_file "${prompt_file}" \
    --seed 42 \
    --resolution 1024 \
    --num_inference_steps 40 \
    --guidance_scale 4.5 \
    --mixed_precision "bf16"

echo "SD-3.5-M long prompt generation completed!"
echo "Images saved to: ${image_dir}/images"
