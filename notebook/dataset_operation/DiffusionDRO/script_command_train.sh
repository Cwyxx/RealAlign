#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=2,3

MASTER_PORT=29501
run_name="top_500_images_pickscore-pick-a-pic-v2"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/${run_name}"
accelerate launch --multi_gpu --num_processes 2 --main_process_port ${MASTER_PORT} train.py \
    --score pickscore \
    --train_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-pick-a-pic-v2 \
    --validation_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-test \
    --run_name ${run_name} \
    --logdir ${output_dir} \
    --batch_size 2 \
    --buffer_size 2 \
    --buffer_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 125 \
    --num_steps 1000 \
    --learning_rate 1e-8 \
    --scale_lr