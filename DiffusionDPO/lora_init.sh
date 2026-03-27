#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=0,1,2,3

beta_dpo=2000
top_N=512
ckpt=1600

unet_init="runwayml/stable-diffusion-v1-5" # "JaydenLu666/InPO-SD1.5" # "runwayml/stable-diffusion-v1-5"
run_name="irl_random_${top_N}_images-no_anime-hpdv3_all-uids-ckpt_1600-dpo_${beta_dpo}"
pretrained_lora_path="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/irl_random_${top_N}_images-no_anime-hpdv3_all-uids/checkpoints/checkpoint-${ckpt}"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/${run_name}"
csv_file_path_train="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/random_512_images-no_anime-hpdv3_all-uids.csv"

echo "top_N: ${top_N}"
echo "run_name: ${run_name}"
echo "output_dir: ${output_dir}"
echo "unet_init: ${unet_init}"
echo "pretrained_lora_path: ${pretrained_lora_path}"
echo "ckpt: ${ckpt}"
echo "beta_dpo: ${beta_dpo}"
echo "csv_file_path_train: ${csv_file_path_train}"

accelerate launch --mixed_precision="fp16"  train-lora_init.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --train_batch_size 2 \
    --dataloader_num_workers 2 \
    --gradient_accumulation_steps 32 \
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
    --csv_file_path_train ${csv_file_path_train} \
    --pretrained_lora_path ${pretrained_lora_path}