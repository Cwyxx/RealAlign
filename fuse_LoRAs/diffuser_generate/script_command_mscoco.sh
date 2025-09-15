#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

model_type="stable_diffusion"
train_caption_dataset="spo_4k"
val_json_data_path="/data_center/data2/dataset/chenwy/21164-data/coco_2014/mscoco_val_2014.json" # "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1/pick_a_pic_validation_prompt_500.json" # val_json_data_path="../dataset/mscoco_val2014.json" "/data_center/data2/dataset/chenwy/21164-data/ffhq/ffhq-10k-captions.json"
image_column="image" # evalset_idx # image
caption_column="text" # text

seed=42
checkpoint=0
global_step=750
finetune_method="spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1 DRaFT_LV-hpsv2
lora_weight_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/${train_caption_dataset}/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
generated_image_output_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}/stable_diffusion_v1_5/${train_caption_dataset}/mscoco_val_2014_10k/${finetune_method}/checkpoint_${checkpoint}_${global_step}"

# runwayml/stable-diffusion-v1-5 CompVis/stable-diffusion-v1-4
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=7 python sd_generate_image.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --lora_weight_dir "${lora_weight_dir}" \
    --output_dir "${generated_image_output_dir}" \
    --val_json_data_path "${val_json_data_path}" \
    --batch_size 8 \
    --seed ${seed} \
    --image_column "${image_column}" \
    --caption_column "${caption_column}"
