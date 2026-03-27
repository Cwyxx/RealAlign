#!/bin/bash

# Forensic-Chat and OmniAID Evaluation for SD-3.5-M
# Usage: bash forensic_chat_eval_sd_3_5_m.sh [cuda_device] [method] [checkpoint] [rl_framework]

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate geneval2

cuda_device=${1:-0}
method=${2:-"sd-3-5-medium"}
ckpt=${3:-0}
rl_framework=${4:-"diffusion-dpo"}

export CUDA_VISIBLE_DEVICES=${cuda_device}
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

# Paths configuration
forensic_model_path="/data_center/data2/dataset/chenwy/21164-data/model-ckpt/Forensic-Chat/Forensic-Chat"
omniaid_ckpt_path="/data_center/data2/dataset/chenwy/21164-data/model-ckpt/OmniAID/OmniAID/router/checkpoint-0.pth"
omniaid_clip_path="openai/clip-vit-large-patch14-336"
effort_ckpt_path="/data_center/data2/dataset/chenwy/21164-data/model-ckpt/Effort/effort_clip_L14_trainOn_sdv14.pth"

base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/generate_images/realgen/${method}/ckpt-${ckpt}"
image_dir="${base_image_dir}"

forensic_output="${base_image_dir}/forensic_chat-results.json"
omniaid_output="${base_image_dir}/omniaid-results.json"
effort_output="${base_image_dir}/effort-results.json"

# Run Forensic-Chat evaluation
echo "Running Forensic-Chat evaluation..."
python ../eval/forensic_chat_eval.py \
    --model_path "${forensic_model_path}" \
    --image_dir "${image_dir}" \
    --output_file "${forensic_output}" \
    --device cuda \
    --mixed_precision bf16

echo "Forensic-Chat evaluation completed!"
echo "Results saved to: ${forensic_output}"
echo ""

# Run OmniAID evaluation
echo "Running OmniAID evaluation..."
python ../eval/omniaid_eval.py \
    --ckpt_path "${omniaid_ckpt_path}" \
    --clip_path "${omniaid_clip_path}" \
    --image_dir "${image_dir}" \
    --output_file "${omniaid_output}" \
    --device cuda

echo "OmniAID evaluation completed!"
echo "Results saved to: ${omniaid_output}"
echo ""

# Run Effort evaluation
echo "Running Effort evaluation..."
python ../eval/effort_eval.py \
    --ckpt_path "${effort_ckpt_path}" \
    --image_dir "${image_dir}" \
    --output_file "${effort_output}" \
    --device cuda

echo "Effort evaluation completed!"
echo "Results saved to: ${effort_output}"
echo ""
echo "All evaluations completed for SD-3.5-M!"

