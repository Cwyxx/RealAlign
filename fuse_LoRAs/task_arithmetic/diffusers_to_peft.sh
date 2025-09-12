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

epoch=0
global_step=500
adapter_name="code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15"
lora_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/${adapter_name}/checkpoint_${epoch}_${global_step}"
peft_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/${adapter_name}/checkpoint_${epoch}_${global_step}"

# runwayml/stable-diffusion-v1-5 CompVis/stable-diffusion-v1-4
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=5 python diffusers_to_peft.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --lora_dir "${lora_dir}" \
    --adapter_name "code-divert_start_step_15" \
    --peft_dir "${peft_dir}"