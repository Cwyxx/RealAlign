#!/bin/bash

seed=42
checkpoint=0
global_step=0
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/runwayml-stable-diffusion-v1-5" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
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

# # VQAScore metric
# echo "********************{VQAScore}********************"
# echo "Running VQAScore metric..."
# ./reward_model_metric_param.sh "vqascore" "${generated_image_dir}" "${val_json_data_path}" "${image_column}" "${caption_column}"


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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model


#### spo_official ####
seed=42
checkpoint=0
global_step=0
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/spo-sdv1-5/spo_official" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model
#### spo_official ####


#### dinov2 ####
seed=42
checkpoint=0
global_step=800
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/spo-sdv1-5/dinov2" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model
#### dinov2 ####


#### dinov2-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15 ####
seed=42
checkpoint=0
global_step=500
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/spo-sdv1-5/dinov2-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model
#### dinov2-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15 ####

#### code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15 ####
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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model
#### code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15 ####

#### fuse_lora/concatenation/step_aware_1.0-dinov2_1.0 ####
seed=42
checkpoint=0
global_step=0
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/spo-sdv1-5/fuse_lora/concatenation/step_aware_1.0-dinov2_1.0" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model
#### fuse_lora/concatenation/step_aware_1.0-dinov2_1.0 ####


#### fuse_lora/concatenation/step_aware_1.0-code_1.0 ####
seed=42
checkpoint=0
global_step=0
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/spo-sdv1-5/fuse_lora/concatenation/step_aware_1.0-code_1.0" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model
#### fuse_lora/concatenation/step_aware_1.0-code_1.0 ####

#### fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_1.0 ####
seed=42
checkpoint=0
global_step=0
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/spo-sdv1-5/fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_1.0" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model
#### fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_1.0 ####

#### fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_2.0 ####
seed=42
checkpoint=0
global_step=0
finetune_method="stable_diffusion_v1_5/spo_4k/pick_a_pic_validation_500/spo-sdv1-5/fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_2.0" # DRaFT_LV-adv-5-imagereward-JPEG_1.0_80_100-grad_scale_0.1
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

# VILA metric
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vila
echo "********************{VILA}********************"
echo "Running VILA metric..."
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python3 -m vila.run_vila_predict \
    --image_dir ${generated_image_dir} \
    --ckpt_dir /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/checkpoints/vila_rank_tuned/ \
    --spm_model_path /data_center/data2/dataset/chenwy/21164-data/model-ckpt/vila/spm_model/spm.model
#### fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_2.0 ####