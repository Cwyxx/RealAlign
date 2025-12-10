#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=1,2,3,4

MASTER_PORT=29503
unet_init="runwayml/stable-diffusion-v1-5"
run_name="random_9984_images_no_anime_pickscore_002_ckpt_600-next-irl_top_500_pickscore_images_hpdv3_all"
pretrained_lora_path="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/random_9984_images_no_anime_pickscore_002-hpdv3_all-inpainting/checkpoints/checkpoint-600"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/${run_name}"
accelerate launch --multi_gpu --num_processes 4 --main_process_port ${MASTER_PORT} train-next-irl.py \
    --score pickscore \
    --train_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-top_500_pickscore_images \
    --validation_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-test \
    --run_name ${run_name} \
    --logdir ${output_dir} \
    --num_steps 1600 \
    --unet_init ${unet_init} \
    --pretrained_lora_path ${pretrained_lora_path}