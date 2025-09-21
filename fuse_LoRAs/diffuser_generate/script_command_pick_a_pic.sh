#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate internvl

# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/step_aware/checkpoint_1_754"
# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/dinov2/checkpoint_0_800"
# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/dinov2-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15/checkpoint_0_500"
# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0/checkpoint_0_800"
# "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15/checkpoint_0_750"

model_type="stable_diffusion"
train_caption_dataset="spo_4k"
val_json_data_path="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1/pick_a_pic_validation_prompt_500.json" # "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1/pick_a_pic_validation_prompt_500.json" # val_json_data_path="../dataset/mscoco_val2014.json" "/data_center/data2/dataset/chenwy/21164-data/ffhq/ffhq-10k-captions.json"
image_column="evalset_idx" # evalset_idx # image
caption_column="caption" # text

#### spo_official ####
seed=42
num_inference_steps=8
scheduler="dpmsolver"
checkpoint=0
global_step=0
finetune_method="spo-sdv1-5/spo_official" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1 DRaFT_LV-hpsv2
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/${train_caption_dataset}/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}-dpmsolver/stable_diffusion_v1_5/${train_caption_dataset}/pick_a_pic_validation_500/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

# runwayml/stable-diffusion-v1-5 CompVis/stable-diffusion-v1-4
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 python sd_generate_image.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --lora_weight_dir "${lora_weight_dir}" \
    --output_dir "${generated_image_output_dir}" \
    --val_json_data_path "${val_json_data_path}" \
    --batch_size 4 \
    --seed ${seed} \
    --image_column "${image_column}" \
    --caption_column "${caption_column}" \
    --num_inference_steps ${num_inference_steps} \
    --scheduler ${scheduler}
#### spo_official ####

#### step_aware ####
seed=42
num_inference_steps=8
scheduler="dpmsolver"
checkpoint=1
global_step=754
finetune_method="spo-sdv1-5/step_aware" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1 DRaFT_LV-hpsv2
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/${train_caption_dataset}/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}-dpmsolver/stable_diffusion_v1_5/${train_caption_dataset}/pick_a_pic_validation_500/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

# runwayml/stable-diffusion-v1-5 CompVis/stable-diffusion-v1-4
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 python sd_generate_image.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --lora_weight_dir "${lora_weight_dir}" \
    --output_dir "${generated_image_output_dir}" \
    --val_json_data_path "${val_json_data_path}" \
    --batch_size 4 \
    --seed ${seed} \
    --image_column "${image_column}" \
    --caption_column "${caption_column}" \
    --num_inference_steps ${num_inference_steps} \
    --scheduler ${scheduler}
#### step_aware ####

#### code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15 ####
seed=42
num_inference_steps=8
scheduler="dpmsolver"
checkpoint=0
global_step=750
finetune_method="spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1 DRaFT_LV-hpsv2
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/${train_caption_dataset}/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}-dpmsolver/stable_diffusion_v1_5/${train_caption_dataset}/pick_a_pic_validation_500/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

# runwayml/stable-diffusion-v1-5 CompVis/stable-diffusion-v1-4
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 python sd_generate_image.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --lora_weight_dir "${lora_weight_dir}" \
    --output_dir "${generated_image_output_dir}" \
    --val_json_data_path "${val_json_data_path}" \
    --batch_size 4 \
    --seed ${seed} \
    --image_column "${image_column}" \
    --caption_column "${caption_column}" \
    --num_inference_steps ${num_inference_steps} \
    --scheduler ${scheduler}
#### code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15 ####

#### runwayml-stable-diffusion-v1-5 ####
seed=42
num_inference_steps=8
scheduler="dpmsolver"
checkpoint=0
global_step=0
finetune_method="runwayml-stable-diffusion-v1-5" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1 DRaFT_LV-hpsv2
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/${train_caption_dataset}/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}-dpmsolver/stable_diffusion_v1_5/${train_caption_dataset}/pick_a_pic_validation_500/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

# runwayml/stable-diffusion-v1-5 CompVis/stable-diffusion-v1-4
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 python sd_generate_image.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --lora_weight_dir "${lora_weight_dir}" \
    --output_dir "${generated_image_output_dir}" \
    --val_json_data_path "${val_json_data_path}" \
    --batch_size 4 \
    --seed ${seed} \
    --image_column "${image_column}" \
    --caption_column "${caption_column}" \
    --num_inference_steps ${num_inference_steps} \
    --scheduler ${scheduler}
#### runwayml-stable-diffusion-v1-5 ####