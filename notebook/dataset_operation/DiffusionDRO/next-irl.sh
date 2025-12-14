#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=4,5,6,7

MASTER_PORT=29500

echo "dpo_official -- Inverse Reinforcement Learning"
unet_init="mhdang/dpo-sd1.5-text2image-v1" 
run_name="dpo_official-irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/${run_name}"
accelerate launch --multi_gpu --num_processes 4 --main_process_port ${MASTER_PORT} train-next-irl.py \
    --score pickscore \
    --train_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all \
    --validation_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-test \
    --run_name ${run_name} \
    --logdir ${output_dir} \
    --num_steps 1600 \
    --unet_init ${unet_init}

sleep 300 # 5 minutes
echo "spo_official -- Inverse Reinforcement Learning"
unet_init="runwayml/stable-diffusion-v1-5"
pretrained_lora_path="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/spo-official/checkpoints/checkpoint-0"
run_name="spo_official-irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/${run_name}"
accelerate launch --multi_gpu --num_processes 4 --main_process_port ${MASTER_PORT} train-next-irl.py \
    --score pickscore \
    --train_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all \
    --validation_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-test \
    --run_name ${run_name} \
    --logdir ${output_dir} \
    --num_steps 1600 \
    --unet_init ${unet_init} \
    --pretrained_lora_path ${pretrained_lora_path}