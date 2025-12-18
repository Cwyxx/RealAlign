#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export TOKENIZERS_PARALLELISM=False

cuda_device=$1 # 0
method=$2 # "sd-3-5-medium"
ckpt=$3 # 0
dataset="drawbench_realistic_style"
rl_framework="diffusion-dpo"

export CUDA_VISIBLE_DEVICES=${cuda_device}

base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/model-ckpt"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/generate_images/${dataset}"

seed=42
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"
echo "********************************************"
echo "dataset: ${dataset}"
echo "ckpt_dir: ${ckpt_dir}"
echo "image_dir: ${image_dir}"

python generate_image.py --seed ${seed} --checkpoint_path ${ckpt_dir} --model_type sd3 --dataset ${dataset} \
     --output_dir ${image_dir} \
     --save_images

# reward_model_list=("pickscore" "imagereward" "hpsv3" "aesthetic" "deqa")
# reward_model_list=("pickscore" "imagereward" "aesthetic" "deqa")
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

# conda activate vila
# echo "********************************************"
# echo "reward_model: vila_score"
# cd ../../../evaluate_metric
# python3 -m vila.run_vila_predict_by_gemini_diffusionnft \
#     --output_dir ${image_dir} \
#     --ckpt_dir "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/" \
#     --spm_model_path "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model" \
#     --dataset "${dataset}"

# cd ../DiffusionNFT/scripts/evaluation-sd-3-5-medium

echo "********************************************"
echo "reward_model: MA-AGIQA"
conda activate mplug_owl2
cd ../../../evaluate_metric/MA-AGIQA
python inference_diffusionnft.py --config configs/AGIQA_3k/MA_AGIQA.yaml --dataset ${dataset} --output_dir ${image_dir}
cd ../../DiffusionNFT/scripts/evaluation-sd-3-5-medium

# echo "********************************************"
# echo "reward_model: PKU-AIGIQA"
# conda activate alignprop
# cd ../../../evaluate_metric/PKU-AIGIQA-4K
# python inference_diffusionnft.py --dataset ${dataset} --output_dir ${image_dir}
# cd ../../DiffusionNFT/scripts/evaluation
