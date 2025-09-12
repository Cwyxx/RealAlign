#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 python train_scripts/train_spo.py --config configs/spo_config/spo_reward_aigi_sdv1-5.py
HF_ENDPOINT=https://hf-mirror.com accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_spo.py --config configs/spo_config/spo_sdv1-5.py