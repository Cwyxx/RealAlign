#! /bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

cuda_device=$1 # 0
method=$2 # "sd-3-5-medium"
ckpt=$3 # 0
rl_framework=$4 # "diffusion-dpo"
dataset="wise"

export CUDA_VISIBLE_DEVICES=${cuda_device}
export HF_ENDPOINT=https://hf-mirror.com 
export TOKENIZERS_PARALLELISM=False

json_data_dir="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/benchmark-evaluation/WISE/data"
base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/model-ckpt"
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/generate_images/${dataset}"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"

# json_path 后缀：wise -> xxx.json，wise-rewrite -> xxx_rewrite.json
if [ "${dataset}" = "wise-rewrite" ]; then
  json_suffix="_rewrite.json"
else
  json_suffix=".json"
fi

# # #### Generating ####
# python generate-image-sd-3-5-medium.py \
#     --checkpoint_path "${ckpt_dir}" \
#     --output_dir "${image_dir}" \
#     --json_data_dir "${json_data_dir}" \
#     --seed 42 \
#     --resolution 512 \
#     --mixed_precision "fp16" \
#     --dataset ${dataset}

# #### Evaluating ####
conda activate utils
max_workers=50
api_base="https://api2.aigcbest.top/v1" # "https://35.aigcbest.top/v1" #
private_api_key="sk-SpCI4PAYOSqh8uQ8qf42XGTm8CttGwKGwyrWSdmnZl4GmObD" # "sk-otakhWz5JJkSiR8p1eB61dB6658e410a89Ac2c6bA726Ed89" #
model="gemini-3.1-flash-lite-preview" # "gpt-4o-2024-05-13" # "claude-haiku-4-5-20251001"
python gpt_eval.py \
    --json_path data/cultural_common_sense${json_suffix} \
    --output_dir ${image_dir}/Results-${model}/cultural_common_sense \
    --image_dir ${image_dir}/images \
    --api_key ${private_api_key} \
    --api_base ${api_base} \
    --model "${model}" \
    --result_full ${image_dir}/Results-${model}/cultural_common_sense_full_results.json \
    --result_scores ${image_dir}/Results-${model}/cultural_common_sense_scores_results.jsonl \
    --max_workers ${max_workers}

python gpt_eval.py \
    --json_path data/spatio-temporal_reasoning${json_suffix} \
    --output_dir ${image_dir}/Results-${model}/spatio-temporal_reasoning \
    --image_dir ${image_dir}/images \
    --api_key ${private_api_key} \
    --api_base ${api_base} \
    --model "${model}" \
    --result_full ${image_dir}/Results-${model}/spatio-temporal_reasoning_results.json \
    --result_scores ${image_dir}/Results-${model}/spatio-temporal_reasoning_scores_results.jsonl \
    --max_workers ${max_workers}

python gpt_eval.py \
    --json_path data/natural_science${json_suffix} \
    --output_dir ${image_dir}/Results-${model}/natural_science \
    --image_dir ${image_dir}/images \
    --api_key ${private_api_key} \
    --api_base ${api_base} \
    --model "${model}" \
    --result_full ${image_dir}/Results-${model}/natural_science_full_results.json \
    --result_scores ${image_dir}/Results-${model}/natural_science_scores_results.jsonl \
    --max_workers ${max_workers}

# # # #### Calculating scores ####
python Calculate.py \
    "${image_dir}/Results-${model}/cultural_common_sense_scores_results.jsonl" \
    "${image_dir}/Results-${model}/spatio-temporal_reasoning_scores_results.jsonl" \
    "${image_dir}/Results-${model}/natural_science_scores_results.jsonl" \
    --category all