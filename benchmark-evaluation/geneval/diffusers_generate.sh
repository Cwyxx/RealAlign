#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate internvl

metadata_file="prompts/evaluation_metadata.jsonl"

#### runwayml-stable-diffusion-v1-5 ####
seed=42
num_inference_steps=30
checkpoint=0
global_step=0
finetune_method="runwayml-stable-diffusion-v1-5"
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python generation/diffusers_generate.py \
    ${metadata_file} \
    --model "runwayml/stable-diffusion-v1-5" \
    --outdir "${generated_image_output_dir}" \
    --lora_weight_dir "${lora_weight_dir}" \
    --seed ${seed} \
    --num_inference_steps ${num_inference_steps}
#### runwayml-stable-diffusion-v1-5 ####

#### spo_official ####
seed=42
num_inference_steps=30
checkpoint=0
global_step=0
finetune_method="spo-sdv1-5/spo_official"
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python generation/diffusers_generate.py \
    ${metadata_file} \
    --model "runwayml/stable-diffusion-v1-5" \
    --outdir "${generated_image_output_dir}" \
    --lora_weight_dir "${lora_weight_dir}" \
    --seed ${seed} \
    --num_inference_steps ${num_inference_steps}
#### spo_official ####

#### step_aware ####
seed=42
num_inference_steps=30
checkpoint=1
global_step=754
finetune_method="spo-sdv1-5/step_aware"
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python generation/diffusers_generate.py \
    ${metadata_file} \
    --model "runwayml/stable-diffusion-v1-5" \
    --outdir "${generated_image_output_dir}" \
    --lora_weight_dir "${lora_weight_dir}" \
    --seed ${seed} \
    --num_inference_steps ${num_inference_steps}
#### step_aware ####

#### code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15 ####
seed=42
num_inference_steps=30
checkpoint=0
global_step=750
finetune_method="spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15"
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python generation/diffusers_generate.py \
    ${metadata_file} \
    --model "runwayml/stable-diffusion-v1-5" \
    --outdir "${generated_image_output_dir}" \
    --lora_weight_dir "${lora_weight_dir}" \
    --seed ${seed} \
    --num_inference_steps ${num_inference_steps}
#### code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15 ####