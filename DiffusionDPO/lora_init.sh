#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=6,7

top_N=256
unet_init="runwayml/stable-diffusion-v1-5"
run_name="irl_top_${top_N}_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_ckpt_1600-dpo_top_${top_N}_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_inpainting"
pretrained_lora_path="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/irl_top_${top_N}_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all/checkpoints/checkpoint-1600"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/${run_name}"

echo "top_N: ${top_N}"
echo "run_name: ${run_name}"
echo "output_dir: ${output_dir}"
echo "unet_init: ${unet_init}"
echo "pretrained_lora_path: ${pretrained_lora_path}"

accelerate launch --mixed_precision="fp16"  train-lora_init.py --pretrained_model_name_or_path ${unet_init} \
    --train_batch_size 2 \
    --dataloader_num_workers 2 \
    --gradient_accumulation_steps 64 \
    --max_train_steps 1000 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 125 \
    --learning_rate 1e-8 --scale_lr \
    --checkpointing_steps 100 \
    --beta_dpo 5000 \
    --output_dir ${output_dir} \
    --run_name ${run_name} \
    --pretrained_lora_path ${pretrained_lora_path} \
    --unet_init ${unet_init}