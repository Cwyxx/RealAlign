#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=3,4,5,7
# export HF_HOME_TOKEN="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"
torchrun --nproc_per_node=4 scripts/train_nft_sd3_filter_sample.py --config config/nft.py:sd3_code
