#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export TOKENIZERS_PARALLELISM=False

cuda_device=$1 # 0
method=$2 # "sd-3-5-medium"
ckpt=$3 # 0
rl_framework=$4 # "diffusion-dpo"
dataset=$5 # "pick_a_pic_v2"

export CUDA_VISIBLE_DEVICES=${cuda_device}

base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/model-ckpt"

seed_list=(123 456 789 1000)
for seed in "${seed_list[@]}"; do
    conda activate alignprop
    echo "********************************************"
    echo "Starting evaluation with seed: ${seed}"
    base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/generate_images_seed_${seed}/${dataset}"
    ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
    image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"

    if [[ "$method" == "dpo-official" ]] || [[ "$method" == "kto-official" ]] || [[ "$method" == "sd-v1-5" ]]; then
        if [[ "$method" == "dpo-official" ]]; then
            unet_init="mhdang/dpo-sd1.5-text2image-v1"
        elif [[ "$method" == "kto-official" ]]; then
            unet_init="jacklishufan/diffusion-kto"
        elif [[ "$method" == "sd-v1-5" ]]; then
            unet_init="runwayml/stable-diffusion-v1-5"
        fi
        python generate_image-unet_init.py --seed ${seed} --dataset ${dataset} \
            --output_dir ${image_dir} \
            --save_images \
            --unet_init ${unet_init}
    else
        python generate_image.py --seed ${seed} --checkpoint_path ${ckpt_dir} --dataset ${dataset} \
            --output_dir ${image_dir} \
            --save_images
    fi

    cd ../evaluation-sd-3-5-medium
    reward_model_list=("pickscore" "imagereward" "hpsv3" "deqa" "aesthetic" "unifiedreward")
    for reward_model in "${reward_model_list[@]}"; do
        echo "********************************************"
        echo "reward_model: ${reward_model}"
        conda activate alignprop

        if [[ "$reward_model" == "deqa" ]] || [[ "$reward_model" == "clip_iqa" ]] || [[ "$reward_model" == "q-align" ]]; then
            conda activate internvl
        elif [[ "$reward_model" == "aesthetic_v2_5" ]] || [[ "$reward_model" == "unifiedreward" ]]; then
            conda activate utils
        elif [[ "$reward_model" == "vqascore" ]]; then
            conda activate t2v
        elif [[ "$reward_model" == "hpsv3" ]] || [[ "$reward_model" == "SGP-HPSv3" ]]; then
            conda activate hpsv3
        elif [[ "$reward_model" == "cpbd" ]]; then
            conda activate utils
        elif [[ "$reward_model" == "imagedoctor" ]] || [[ "$reward_model" == "diffdoctor" ]]; then
            conda activate imagedoctor
        fi

        python calculate_score.py --reward_model ${reward_model} --dataset ${dataset} --output_dir ${image_dir} 
    done
    cd ../evaluation-sd-v1-5

    # echo "********************************************"
    # echo "reward_model: MA-AGIQA"
    # conda activate mplug_owl2
    # cd ../../../evaluate_metric/MA-AGIQA
    # python inference_diffusionnft.py --config configs/AGIQA_3k/MA_AGIQA.yaml --dataset ${dataset} --output_dir ${image_dir}
    # cd ../../DiffusionNFT/scripts/evaluation-sd-3-5-medium
done