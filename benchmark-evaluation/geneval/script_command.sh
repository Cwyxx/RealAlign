#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

cuda_device=$1 # 0
method=$2 # "sd-3-5-medium"
ckpt=$3 # 0
rl_framework="diffusion-dpo"
dataset="geneval"

metadata_file="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/geneval/prompts/evaluation_metadata.jsonl"
base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/model-ckpt"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/generate_images/${dataset}"
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"

echo "********************************************"
echo "dataset: ${dataset}"
echo "ckpt_dir: ${ckpt_dir}"
echo "image_dir: ${image_dir}"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=${cuda_device} python generation/sd_3_5_medium/diffusers_generate.py \
    ${metadata_file} \
    --model "stabilityai/stable-diffusion-3.5-medium" \
    --outdir "${image_dir}" \
    --checkpoint_path "${ckpt_dir}" \
    --seed 42 \
    --num_inference_steps 40 \
    --scale 4.5 \
    --batch_size 1 \
    --mixed_precision "fp16" \

conda activate internvl
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation/evaluate_images.py \
    "${image_dir}" \
    --outfile "${image_dir}/results.jsonl" \
    --model-path "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/geneval"

echo "summary_scores"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation/summary_scores.py "${image_dir}/results.jsonl"
