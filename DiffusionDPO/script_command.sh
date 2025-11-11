#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=3,6

run_name="top_512_images-pickscore-0.02-hpdv3-inpainting"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/${run_name}"
accelerate launch --mixed_precision="fp16"  train-lora.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --train_batch_size 2 \
    --dataloader_num_workers 2 \
    --gradient_accumulation_steps 64 \
    --max_train_steps 1000 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 125 \
    --learning_rate 1e-8 --scale_lr \
    --checkpointing_steps 50 \
    --beta_dpo 5000 \
    --output_dir ${output_dir} \
    --run_name ${run_name}