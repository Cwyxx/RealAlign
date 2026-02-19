#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com

prompt_file="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/no_anime_colorfulness-hpdv3_all-uids.csv"
input_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/HPDv3/real"
output_dir="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net-next-sd3.5m-inpainting/HPDv3"

GPU_ID="2 3 6 7"
CHUNK=2750
gpu_list=(${GPU_ID})
for i in 0 1 2 3; do
  start_index=$((i * CHUNK))
  end_index=$(((i + 1) * CHUNK))
  CUDA_VISIBLE_DEVICES=${gpu_list[i]} python u2net-next-sd3.5m-inpainting.py \
    --input_dir "${input_dir}" --output_dir "${output_dir}" --prompt_file "${prompt_file}" \
    --start_index ${start_index} --end_index ${end_index} &
  sleep 60

done
wait
echo "All 4 processes finished."