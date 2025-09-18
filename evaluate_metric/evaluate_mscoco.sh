#!/bin/bash

seed=42
checkpoint=1
global_step=754
finetune_method="stable_diffusion_v1_5/spo_4k/mscoco_val_2014_10k/spo-sdv1-5/step_aware" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
generated_image_dir="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}/${finetune_method}/checkpoint_${checkpoint}_${global_step}" # "/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_1/stable_diffusion_v1_4/spo_4k/${finetune_method}/checkpoint_${checkpoint}_${global_step}"
val_json_data_path="/data_center/data2/dataset/chenwy/21164-data/coco_2014/mscoco_val_2014.json" # "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1/pick_a_pic_validation_prompt_500.json" # "../../dataset/mscoco_val2014.json" # "/data_center/data2/dataset/chenwy/21164-data/ffhq/ffhq-10k-captions.json"
reference_image_dir="/data_center/data2/dataset/chenwy/21164-data/coco_2014/val2014-10k"
image_column="image" # "evalset_idx" "image"
caption_column="text" # "caption" "text"
echo "******************************************************"
echo "validation dataset: mscoco_2014_val_10k"
echo "seed: ${seed}"
echo "checkpoint: ${checkpoint}"
echo "global_step: ${global_step}"
echo "finetune method: ${finetune_method}"
echo "reference_image_dir: ${reference_image_dir}"
echo "generated image dir: ${generated_image_dir}"
echo "val json data path: ${val_json_data_path}"

# # Clean FID metric
# echo "********************{Clean FID}********************"
# echo "Running Clean FID metric..."
# cd clean_fid
# ./clean_fid_param.sh "Clean-FID" "${reference_image_dir}" "${generated_image_dir}"
# cd ..

# # CMMD metric
# cd cmmd-pytorch
# echo "********************{CMMD}********************"
# echo "Running CMMD metric..."
# ./cmmd_param.sh "${reference_image_dir}" "${generated_image_dir}"
# cd ..

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

# # VQAScore metric
# echo "********************{VQAScore}********************"
# echo "Running VQAScore metric..."
# ./reward_model_metric_param.sh "vqascore" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"

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

# # Aesthetic-V2-5 metric
# echo "********************{Aesthetic_V2_5}********************"
# echo "Running Aesthetic_V2_5 metric..."
# ./reward_model_metric_param.sh "aesthetic_v2_5" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"

# # VILA metric
# source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
# conda activate vila
# echo "********************{VILA}********************"
# echo "Running VILA metric..."
# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python -m vila.run_vilda_predict.py \
#     -ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints \
#     --image_dir ${generated_image_dir} \
#     --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model