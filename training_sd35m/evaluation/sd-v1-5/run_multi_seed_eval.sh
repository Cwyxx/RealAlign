#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False
export TMPDIR=/data1/chenwy/newtmp

cuda_device=$1 # 0
method=$2 # "sd-v1-5"
ckpt=$3 # 0
rl_framework=$4 # "diffusion-dpo"
dataset=$5 # "pick_a_pic_v2"

export CUDA_VISIBLE_DEVICES=${cuda_device}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/model-ckpt"

seed_list=(42 123 456 789 1000)
for seed in "${seed_list[@]}"; do
    echo "********************************************"
    echo "Starting evaluation with seed: ${seed}"
    base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/generate_images_seed_${seed}/${dataset}"
    image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"
    unet_init="runwayml/stable-diffusion-v1-5"
    checkpoint_args=()

    if [[ "$method" == *"dpo_official-"* ]]; then
        unet_init="mhdang/dpo-sd1.5-text2image-v1"
        checkpoint_args=(--checkpoint_path "${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}")
    elif [[ "$method" == "dpo-official" ]]; then
        unet_init="mhdang/dpo-sd1.5-text2image-v1"
    elif [[ "$method" != "sd-v1-5" ]]; then
        checkpoint_args=(--checkpoint_path "${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}")
    fi

    conda activate alignprop
    python "${script_dir}/generate_image.py" \
        --seed "${seed}" \
        --dataset "${dataset}" \
        --unet_init "${unet_init}" \
        "${checkpoint_args[@]}" \
        --output_dir "${image_dir}" \
        --save_images

    reward_model_list=("pickscore" "imagereward" "hpsv3" "aesthetic" "deqa" "unifiedreward")
    for reward_model in "${reward_model_list[@]}"; do
        echo "********************************************"
        echo "reward_model: ${reward_model}"
        conda activate alignprop

        if [[ "$reward_model" == "deqa" ]]; then
            conda activate internvl
        elif [[ "$reward_model" == "unifiedreward" ]]; then
            conda activate utils
        elif [[ "$reward_model" == "hpsv3" ]]; then
            conda activate hpsv3
        fi

        python "${script_dir}/calculate_score.py" --reward_model "${reward_model}" --dataset "${dataset}" --output_dir "${image_dir}"
    done
done
