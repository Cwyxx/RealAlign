#!/bin/bash

seed=42
checkpoint=0
global_step=750
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
generated_image_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}/${finetune_method}/checkpoint_${checkpoint}_${global_step}" # "/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_1/stable_diffusion_v1_4/spo_4k/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
val_json_data_path="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1/pick_a_pic_validation_prompt_500.json" # "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1/pick_a_pic_validation_prompt_500.json" # "../../dataset/mscoco_val2014.json" # "/data_center/data2/dataset/chenwy/21164-data/ffhq/ffhq-10k-captions.json"
image_column="evalset_idx" # "evalset_idx" "image"
caption_column="caption" # "caption" "text"
echo "******************************************************"
echo "validation dataset: pick_a_pic_validation_500"
echo "seed: ${seed}"
echo "checkpoint: ${checkpoint}"
echo "global_step: ${global_step}"
echo "finetune method: ${finetune_method}"
echo "generated image dir: ${generated_image_dir}"
echo "val json data path: ${val_json_data_path}"


# # PickScore metric
# echo "********************{PickScore}********************"
# echo "Running PickScore metric..."
# ./reward_model_metric_param.sh "pickscore" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"

# # HPSv2 metric
# echo "********************{HPSv2}********************"
# echo "Running HPSv2 metric..."
# ./reward_model_metric_param.sh "hpsv2" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"

# # ImageReward metric
# echo "********************{ImageReward}********************"
# echo "Running ImageReward metric..."
# ./reward_model_metric_param.sh "imagereward" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"

# # CLIPScore metric
# echo "********************{CLIPScore}********************"
# echo "Running CLIPScore metric..."
# ./reward_model_metric_param.sh "clipscore" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"

# # CLIP-IQA metric
# echo "********************{CLIP-IQA}********************"
# echo "Running CLIP-IQA metric..."
# ./reward_model_metric_param.sh "clip_iqa" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"


# # DeQA metric
# echo "********************{DeQA}********************"
# echo "Running DeQA metric..."
# ./reward_model_metric_param.sh "deqa" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"

# # Aesthetic metric
# echo "********************{Aesthetic}********************"
# echo "Running Aesthetic metric..."
# ./reward_model_metric_param.sh "aesthetic" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"

# Aesthetic-V2-5 metric
echo "********************{Aesthetic_V2_5}********************"
echo "Running Aesthetic_V2_5 metric..."
./reward_model_metric_param.sh "aesthetic_v2_5" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"
