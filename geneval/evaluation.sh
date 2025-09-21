#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate internvl

seed=42
num_inference_steps=30
epoch=0
global_step=0
finetune_method="spo-sdv1-5/spo_official/checkpoint_${epoch}_${global_step}"
image_folder="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}"
echo "******************************************************"
echo "evaluate_images"
echo "finetune_method: ${finetune_method}"
echo "image_folder: ${image_folder}"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/evaluate_images.py \
    "${image_folder}" \
    --outfile "${image_folder}/results.jsonl" \
    --model-path "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/geneval"


echo "summary_scores"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/summary_scores.py "${image_folder}/results.jsonl"


##### step_aware ####
seed=42
num_inference_steps=30
epoch=1
global_step=754
finetune_method="spo-sdv1-5/step_aware/checkpoint_${epoch}_${global_step}"
image_folder="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}"
echo "******************************************************"
echo "evaluate_images"
echo "finetune_method: ${finetune_method}"
echo "image_folder: ${image_folder}"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/evaluate_images.py \
    "${image_folder}" \
    --outfile "${image_folder}/results.jsonl" \
    --model-path "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/geneval"


echo "summary_scores"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/summary_scores.py "${image_folder}/results.jsonl"
##### step_aware ####


##### code ####
seed=42
num_inference_steps=30
epoch=0
global_step=750
finetune_method="spo-sdv1-5/code-lr_6e-05-max_gn_1.0-comp_0.0-divert_start_step_15/checkpoint_${epoch}_${global_step}"
image_folder="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}"
echo "******************************************************"
echo "evaluate_images"
echo "finetune_method: ${finetune_method}"
echo "image_folder: ${image_folder}"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/evaluate_images.py \
    "${image_folder}" \
    --outfile "${image_folder}/results.jsonl" \
    --model-path "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/geneval"


echo "summary_scores"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/summary_scores.py "${image_folder}/results.jsonl"
##### code ####


##### fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_1.0 ####
seed=42
num_inference_steps=30
epoch=0
global_step=0
finetune_method="spo-sdv1-5/fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_1.0/checkpoint_${epoch}_${global_step}"
image_folder="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_${seed}-num_inference_steps_${num_inference_steps}/stable_diffusion_v1_5/spo_4k/geneval/${finetune_method}"
echo "******************************************************"
echo "evaluate_images"
echo "finetune_method: ${finetune_method}"
echo "image_folder: ${image_folder}"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/evaluate_images.py \
    "${image_folder}" \
    --outfile "${image_folder}/results.jsonl" \
    --model-path "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/geneval"


echo "summary_scores"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python evaluation/summary_scores.py "${image_folder}/results.jsonl"
##### fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_1.0 ####