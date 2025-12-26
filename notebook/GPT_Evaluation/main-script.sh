#! /bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate utils

dataset="OneIG-Bench-Portrait"
txt_path="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT/dataset/${dataset}/test.txt"
image1_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/generate_images/${dataset}/irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all/ckpt-450/images"
image2_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/generate_images/${dataset}/FlowGRPO-PickScore/ckpt-0/images"
output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/GPT_Evaluation_v2/${dataset}/vs-irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all"
model="claude-sonnet-4-5-20250929"

python main.py --txt_path ${txt_path} --model ${model} --image1_dir ${image1_dir} --image2_dir ${image2_dir} --output_dir ${output_dir}