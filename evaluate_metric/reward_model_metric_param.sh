#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

reward_model="$1"
generated_image_dir="$2"
val_json_data_path="$3"
image_column="$4"
caption_column="$5"

if [[ "$reward_model" == "deqa" ]] || [[ "$reward_model" == "clip_iqa" ]] ; then
    conda activate internvl

elif [[ "$reward_model" == "aesthetic_v2_5" ]]; then
    conda activate utils

elif [[ "$reward_model" == "imagereward" ]]; then
    conda activate imagereward
fi


HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=5 python calculate_metric.py \
    --reward_model $reward_model \
    --generated_image_dir $generated_image_dir \
    --val_json_data_path $val_json_data_path \
    --image_column $image_column \
    --caption_column $caption_column    