#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate utils

metric="$1"
reference_image_dir="$2"
generated_image_dir="$3"

CUDA_VISIBLE_DEVICES=5 python clean_fid.py \
    --metric "$metric" \
    --reference_image_dir "$reference_image_dir" \
    --generated_image_dir "$generated_image_dir"