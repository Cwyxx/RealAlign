#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export TOKENIZERS_PARALLELISM=False

cuda_device=$1 # 0
method=$2 # "sd-3-5-medium"
ckpt=$3 # 0
cfg_guidance=$4
dataset="geneval"

export CUDA_VISIBLE_DEVICES=${cuda_device}

num_inference_steps=10
base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images_num_inference_steps_${num_inference_steps}/sd3_textencoder_3_none_cfg_${cfg_guidance}/${dataset}"

ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"
echo "********************************************"
echo "dataset: ${dataset}"
echo "ckpt_dir: ${ckpt_dir}"
echo "image_dir: ${image_dir}"

python generate_image_group.py --seed 42 --checkpoint_path ${ckpt_dir} --model_type sd3 --dataset ${dataset} \
    --output_dir ${image_dir} \
    --guidance_scale ${cfg_guidance} \
    --num_inference_steps ${num_inference_steps} \
    --save_images
