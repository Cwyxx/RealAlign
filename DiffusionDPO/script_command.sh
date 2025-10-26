#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=4,5,6,7

run_name="x_aigd"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/${run_name}"
accelerate launch --mixed_precision="fp16"  train_x_aigd.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --train_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 128 \
    --max_train_steps 1000 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 250 \
    --learning_rate 1e-8 --scale_lr \
    --checkpointing_steps 50 \
    --beta_dpo 5000 \
    --gradient_checkpointing \
    --output_dir ${output_dir} \
    --run_name ${run_name}
