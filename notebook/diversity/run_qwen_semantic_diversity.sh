#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate geneval2

export HF_ENDPOINT=https://hf-mirror.com 
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=4,5,6,7
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSV="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all-uids.csv"
OUTPUT="${SCRIPT_DIR}/results/top_512_pickscore_real_image_categories.csv"   # leave empty to use default: <csv_dir>/diversity_evaluation_results.csv
MODEL="Qwen/Qwen3.5-27B"
MAX_NEW_TOKENS=32

python "${SCRIPT_DIR}/qwen_semantic_diversity_evaluation.py" \
    --csv "$CSV" \
    ${OUTPUT:+--output "$OUTPUT"} \
    --model "$MODEL" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee "${SCRIPT_DIR}/diversity_evaluation.log"
