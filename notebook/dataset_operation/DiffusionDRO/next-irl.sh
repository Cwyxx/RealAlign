#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=4,5,6,7

MASTER_PORT=29500
top_N=512
learning_rate=1e-4
echo "IRL-saliency_inpainting-full_inpainting-str_0.8-cfg_3.0_top_${top_N}_images_NAC_pickscore_0.02-hpdv3_all -- Inverse Reinforcement Learning"
unet_init="runwayml/stable-diffusion-v1-5" 
run_name="irl_saliency_inpainting-full_inpainting-str_0.8-cfg_3.0_top_${top_N}_images_NAC_pickscore_0.02-hpdv3_all-dpo_2000_full_inpainting-str_0.8-cfg_3.0-irl"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/${run_name}"
train_dataset="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-saliency_inpainting-full_inpainting-strength_0.8-guidance_scale_3.0-top_${top_N}_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all"
pretrained_lora_path="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/model-ckpt/irl_saliency_inpainting-full_inpainting-str_0.8-cfg_3.0_top_512_images_NAC_pickscore_002-hpdv3_all-uids_ckpt_1600-dpo_2000_full_inpainting-str_0.8-cfg_3.0/checkpoints/checkpoint-800"

echo "train_dataset: ${train_dataset}"
echo "run_name: ${run_name}"
echo "output_dir: ${output_dir}"
echo "unet_init: ${unet_init}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "top_N: ${top_N}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "learning_rate: ${learning_rate}"
echo "pretrained lora path: ${pretrained_lora_path}"

accelerate launch --multi_gpu --num_processes 4 --main_process_port ${MASTER_PORT} train-next-irl.py \
    --score pickscore \
    --train_dataset ${train_dataset} \
    --validation_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-test \
    --run_name ${run_name} \
    --logdir ${output_dir} \
    --num_steps 1600 \
    --unet_init ${unet_init} \
    --learning_rate ${learning_rate} \
    --pretrained_lora_path ${pretrained_lora_path}
