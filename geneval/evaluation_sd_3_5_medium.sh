#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate internvl

cuda_device=$1 # 0
method=$2 # "sd-3-5-medium"
ckpt=$3 # 0
cfg_guidance=$4
dataset="geneval"

base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3_textencoder_3_none_cfg_${cfg_guidance}/${dataset}"
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"

echo "********************************************"
echo "dataset: ${dataset}"
echo "ckpt_dir: ${ckpt_dir}"
echo "image_dir: ${image_dir}"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/evaluate_images.py \
    "${image_folder}" \
    --outfile "${image_folder}/results.jsonl" \
    --model-path "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/geneval"


echo "summary_scores"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/summary_scores.py "${image_folder}/results.jsonl"
