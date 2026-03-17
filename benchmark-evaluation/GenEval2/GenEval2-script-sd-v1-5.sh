#! /bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

cuda_device=$1 # 0
method=$2 # "sd-v1-5"
ckpt=$3 # 0
rl_framework=$4 # "diffusion-dpo"

export CUDA_VISIBLE_DEVICES=${cuda_device}
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

benchmark_data="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/benchmark-evaluation/GenEval2/geneval2_data.jsonl"
base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/model-ckpt"
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/generate_images/geneval2"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"
unet_init="runwayml/stable-diffusion-v1-5"

if [[ "$method" == *"dpo-official"* ]]; then
    unet_init="mhdang/dpo-sd1.5-text2image-v1"
fi

mapping_file="${image_dir}/prompt_image_mapping.json"
scores_file="${image_dir}/scores.json"

# #### Generating ####
python generate-image-sd-v1-5.py \
    --checkpoint_path "${ckpt_dir}" \
    --output_dir "${image_dir}" \
    --benchmark_data "${benchmark_data}" \
    --seed 42 \
    --resolution 512 \
    --unet_init ${unet_init}

#### Evaluation ####
conda activate geneval2
python evaluation.py \
    --benchmark_data "${benchmark_data}" \
    --image_filepath_data "${mapping_file}" \
    --method soft_tifa_gm \
    --output_file "${scores_file}"

#### Soft-TIFA Analysis ####
python soft_tifa_analysis.py \
    --benchmark_data "${benchmark_data}" \
    --score_data "${scores_file}"
