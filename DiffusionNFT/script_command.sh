#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HUGGINGFACE_TOKEN="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"
torchrun --nproc_per_node=4 scripts/train_nft_sd3.py --config config/nft.py:sd3_multi_reward_hpsv2_1