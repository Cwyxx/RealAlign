#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate internvl

epoch=0
global_step=0
finetune_method="fuse_lora/concatenation/step_aware_1.0-code-divert_start_step_15_2.0/checkpoint_${epoch}_${global_step}"
image_folder="/data_center/data2/dataset/chenwy/21164-data/generated_image-seed_42/stable_diffusion_v1_5/spo_4k/geneval/spo-sdv1-5/${finetune_method}"
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
