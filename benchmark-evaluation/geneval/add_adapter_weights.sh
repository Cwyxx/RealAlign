#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate internvl

# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/step_aware/checkpoint_1_754"
# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/dinov2/checkpoint_0_800"
# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/dinov2-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15/checkpoint_0_500"
# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/code-lr_6e-05-max_gn_1.0-comp_0.0/checkpoint_0_800"
# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15/checkpoint_0_750"
metadata_file="prompts/evaluation_metadata.jsonl"

#### ties_svd/step_aware_1.0-dinov2-divert_start_step_15_1.0 #### 
lora_dirs=(
    "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/step_aware/checkpoint_1_754"
    "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/dinov2-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15/checkpoint_0_500"
)
lora_weights=(1.0 1.0)
adapter_names=("step_aware" "dinov2-divert_start_step_15")

seed=42
checkpoint=0
global_step=0
combination_type="ties_svd"
finetune_method="spo-sdv1-5/fuse_lora/${combination_type}/step_aware_1.0-dinov2-divert_start_step_15_1.0"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate_add_adapter_weights.py \
    ${metadata_file} \
    --model "runwayml/stable-diffusion-v1-5" \
    --outdir "${generated_image_output_dir}" \
    --lora_dirs "${lora_dirs[@]}" \
    --lora_weights "${lora_weights[@]}" \
    --adapter_names "${adapter_names[@]}" \
    --combination_type "${combination_type}" \
    --seed ${seed}

#### ties_svd/step_aware_1.0-dinov2-divert_start_step_15_1.0 #### 

# #### ties_svd/step_aware_1.0-code-divert_start_step_15_1.0 #### 
lora_dirs=(
    "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/step_aware/checkpoint_1_754"
    "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15/checkpoint_0_750"
)
lora_weights=(1.0 1.0)
adapter_names=("step_aware" "code-divert_start_step_15")

seed=42
checkpoint=0
global_step=0
combination_type="ties_svd"
finetune_method="spo-sdv1-5/fuse_lora/${combination_type}/step_aware_1.0-code-divert_start_step_15_1.0"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate_add_adapter_weights.py \
    ${metadata_file} \
    --model "runwayml/stable-diffusion-v1-5" \
    --outdir "${generated_image_output_dir}" \
    --lora_dirs "${lora_dirs[@]}" \
    --lora_weights "${lora_weights[@]}" \
    --adapter_names "${adapter_names[@]}" \
    --combination_type "${combination_type}" \
    --seed ${seed}

# #### ties_svd/step_aware_1.0-code-divert_start_step_15_1.0 #### 