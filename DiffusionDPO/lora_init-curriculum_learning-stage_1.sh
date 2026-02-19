#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=0,1

beta_dpo=2000
top_N=512
ckpt=1600

unet_init="runwayml/stable-diffusion-v1-5" # "JaydenLu666/InPO-SD1.5" # "runwayml/stable-diffusion-v1-5"
run_name="irl_saliency_inpainting-text_to_image_top_${top_N}_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all-uids_lr_1e-4_${ckpt}-dpo_${beta_dpo}_text_to_image_250-dpo_${beta_dpo}_saliency_inpainting"
pretrained_lora_path="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/irl_saliency_inpainting-text_to_image_top_${top_N}_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all-uids_lr_1e-4_1600-dpo_2000_text_to_image/checkpoints/checkpoint-250"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/${run_name}"
dataset_type="hpdv3_all-two_stage_saliency_inpainting"

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
    --max_train_steps 300 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 125 \
    --learning_rate 1e-8 --scale_lr \
    --checkpointing_steps 50 \
    --beta_dpo ${beta_dpo} \
    --output_dir ${output_dir} \
    --run_name ${run_name} \
    --unet_init ${unet_init} \
    --top_N ${top_N} \
    --dataset_type ${dataset_type} \
    --pretrained_lora_path ${pretrained_lora_path}