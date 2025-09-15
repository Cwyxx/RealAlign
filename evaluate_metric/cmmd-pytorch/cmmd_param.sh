#!/bin/bash

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate utils

reference_image_dir="$1"
generated_image_dir="$2"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=5 \
    python main.py "${reference_image_dir}" "${generated_image_dir}"