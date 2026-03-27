#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=4,5,6,7

MASTER_PORT=29502
top_N=512
learning_rate=1e-4
echo "IRL_random_${top_N}_images-no_anime-colorfulness-pickscore_0.02-hpdv3_all -- Inverse Reinforcement Learning"
unet_init="runwayml/stable-diffusion-v1-5"  # "runwayml/stable-diffusion-v1-5" 
run_name="irl_random_${top_N}_images-no_anime-colorfulness-pickscore_0.02-hpdv3_all-uids"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-v1-5/model-ckpt/${run_name}"
train_dataset="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-random_${top_N}_images-no_anime-colorfulness-pickscore_0.02-hpdv3_all-uids"

echo "train_dataset: ${train_dataset}"
echo "run_name: ${run_name}"
echo "output_dir: ${output_dir}"
echo "unet_init: ${unet_init}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "top_N: ${top_N}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "learning_rate: ${learning_rate}"

accelerate launch --multi_gpu --num_processes 4 --main_process_port ${MASTER_PORT} train-next-irl.py \
    --score pickscore \
    --train_dataset ${train_dataset} \
    --validation_dataset /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/DiffusionDRO-HPDv3-test \
    --run_name ${run_name} \
    --logdir ${output_dir} \
    --num_steps 1600 \
    --unet_init ${unet_init} \
    --learning_rate ${learning_rate}
