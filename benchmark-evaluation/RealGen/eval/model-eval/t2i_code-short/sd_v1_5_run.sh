#!/bin/bash

# SD-v1.5 RealBench Short Prompt Generation Script
# Usage: bash sd_v1_5_run.sh [cuda_device] [method] [checkpoint] [rl_framework]

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

cuda_device=${1:-0}
method=${2:-"sd-v1-5"}
ckpt=${3:-0}
rl_framework=${4:-"diffusion-dpo"}

export CUDA_VISIBLE_DEVICES=${cuda_device}
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

# Paths configuration
prompt_file="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/benchmark-evaluation/RealGen/eval/prompt.txt"
base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/model-ckpt"
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/generate_images/realgen"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"
unet_init="runwayml/stable-diffusion-v1-5"

# Set unet_init based on method
if [[ "$method" == *"dpo-official"* ]]; then
    unet_init="mhdang/dpo-sd1.5-text2image-v1"
fi

# Generate images
python sd_v1_5_t2i.py \
    --checkpoint_path "${ckpt_dir}" \
    --output_dir "${image_dir}" \
    --prompt_file "${prompt_file}" \
    --seed 42 \
    --resolution 512 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --unet_init ${unet_init}

echo "SD-v1.5 short prompt generation completed!"
echo "Images saved to: ${image_dir}/images"
