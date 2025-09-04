#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

# lora_dirs=(
#     "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/step_aware/checkpoint_1_754"
#     "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0/checkpoint_0_800"
#     "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/dinov2/checkpoint_0_800"
#     "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/drct_clip-sdv14/checkpoint_0_800"
#     )
# lora_weights=(0.5 0.5)
# adapter_names=("step_aware" "dinov2")

lora_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0/checkpoint_0_800"
adapter_name="code"
peft_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/code-lr_6e-05-max_gn_1.0-comp_0.0/checkpoint_0_800"

# runwayml/stable-diffusion-v1-5 CompVis/stable-diffusion-v1-4
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python diffusers_to_peft.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --lora_dir "${lora_dir}" \
    --adapter_name "${adapter_name}" \
    --peft_dir "${peft_dir}"