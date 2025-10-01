#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=7

# pickscore: /data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3/pickscore/checkpoints/checkpoint-150
# code: /data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3/code/checkpoints/checkpoint-2
dataset="pick_a_pic_spo"
combination_type="cat"
method="fuse_lora/${combination_type}/pickscore_1.0-code_1.0"
cfg_guidance=1.0
ckpt=1
base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3_cfg_${cfg_guidance}/${dataset}"

image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"
ckpt_dirs=(
        "/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3/pickscore/checkpoints/checkpoint-150" 
        "/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3/code/checkpoints/checkpoint-1"
        )
lora_weights=(1.0 1.0)
adapter_names=("pickscore" "code")
echo "********************************************"
echo "dataset: ${dataset}"
echo "image_dir: ${image_dir}"

python generate_image_lora_merge.py --seed 42 --checkpoint_paths ${ckpt_dirs[@]} --lora_weights ${lora_weights[@]} --adapter_names ${adapter_names[@]} \
    --combination_type ${combination_type} \
    --model_type sd3 --dataset ${dataset} \
    --output_dir ${image_dir} \
    --guidance_scale ${cfg_guidance} \
    --save_images