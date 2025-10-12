#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export TOKENIZERS_PARALLELISM=False

cuda_device=$1 # 0
method=$2 # "sd-3-5-medium"
ckpt=$3 # 0
cfg_guidance=$4
dataset="drawbench"

export CUDA_VISIBLE_DEVICES=${cuda_device}

base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3_textencoder_3_none_cfg_${cfg_guidance}/${dataset}"

ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"
echo "********************************************"
echo "dataset: ${dataset}"
echo "ckpt_dir: ${ckpt_dir}"
echo "image_dir: ${image_dir}"

# python generate_image.py --seed 42 --checkpoint_path ${ckpt_dir} --model_type sd3 --dataset ${dataset} \
#     --output_dir ${image_dir} \
#     --guidance_scale ${cfg_guidance} \
#     --save_images
# chmod 777 "${image_dir}/evaluation_results.jsonl"

# reward_model_list=("pickscore" "hpsv2" "imagereward" "unifiedreward" "clipscore" "vqascore" "clip_iqa" "deqa" "aesthetic" "aesthetic_v2_5")
reward_model_list=("hpsv3")
for reward_model in "${reward_model_list[@]}"; do
    echo "********************************************"
    echo "reward_model: ${reward_model}"
    conda activate alignprop
    
    if [[ "$reward_model" == "deqa" ]] || [[ "$reward_model" == "clip_iqa" ]]; then
        conda activate internvl
    elif [[ "$reward_model" == "aesthetic_v2_5" ]] || [[ "$reward_model" == "unifiedreward" ]]; then
        conda activate utils
    elif [[ "$reward_model" == "vqascore" ]]; then
        conda activate t2v
    elif [[ "$reward_model" == "hpsv3" ]]; then
        conda activate hpsv3
    fi
    
    python calculate_score.py --reward_model ${reward_model} --dataset ${dataset} --output_dir ${image_dir} 
done

# conda activate vila
# echo "********************************************"
# echo "reward_model: vila_score"
# cd ../../../evaluate_metric
# python3 -m vila.run_vila_predict_by_gemini_diffusionnft \
#     --output_dir ${image_dir} \
#     --ckpt_dir "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/" \
#     --spm_model_path "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model" \
#     --dataset "${dataset}"

# cd ../DiffusionNFT/scripts/evaluation