#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=3,4

beta_dpo=2000
top_N=512
ckpt=1600

unet_init="runwayml/stable-diffusion-v1-5"
run_name="irl_top_${top_N}_images_no_anime_colorfulness_pickscore-hpdv3_all-uids_ckpt_${ckpt}-dpo_${beta_dpo}_top_${top_N}_images_pickscore-hpdv3_all"
pretrained_lora_path="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/irl_top_${top_N}_images_no_anime_colorfulness_pickscore-hpdv3_all-uids_lr_1e-4/checkpoints/checkpoint-${ckpt}"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/${run_name}"
dataset_type="hpdv3_all"

echo "top_N: ${top_N}"
echo "run_name: ${run_name}"
echo "output_dir: ${output_dir}"
echo "unet_init: ${unet_init}"
echo "pretrained_lora_path: ${pretrained_lora_path}"
echo "ckpt: ${ckpt}"
echo "beta_dpo: ${beta_dpo}"

accelerate launch --mixed_precision="fp16"  train-lora_init.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --train_batch_size 2 \
    --dataloader_num_workers 2 \
    --gradient_accumulation_steps 64 \
    --max_train_steps 1000 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 125 \
    --learning_rate 1e-8 --scale_lr \
    --checkpointing_steps 100 \
    --beta_dpo ${beta_dpo} \
    --output_dir ${output_dir} \
    --run_name ${run_name} \
    --unet_init ${unet_init} \
    --top_N ${top_N} \
    --dataset_type ${dataset_type} \
    --pretrained_lora_path ${pretrained_lora_path}