#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

export HF_ENDPOINT=https://hf-mirror.com 
export TOKENIZERS_PARALLELISM=False

cuda_device_list=(0 1 2 3 4 5 6 7)
rl_framework="diffusion-dpo"
dataset="coco"
start_index_list=(0 1250 2500 3750 5000 6250 7500 8750)
end_index_list=(1250 2500 3750 5000 6250 7500 8750 10000)
method_list=("FlowGRPO-PickScore" "sd-3-5-medium" "DiffusionNFT")
cfg_guidance_list=(4.5 5.0 4.5)
ckpt_list=(0 0 0)

base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/model-ckpt"
base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-3-5-medium/generate_images/${dataset}"

for method_idx in "${!method_list[@]}"; do
    method="${method_list[$method_idx]}"
    cfg_guidance="${cfg_guidance_list[$method_idx]}"
    ckpt="${ckpt_list[$method_idx]}"
    
    echo "=========================================="
    echo "Starting method: ${method}"
    echo "CFG Guidance: ${cfg_guidance}"
    echo "Checkpoint: ${ckpt}"
    echo "=========================================="
    
    # 构建 checkpoint 路径
    ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"
    
    # 启动 5 个并行任务
    pids=()
    for i in "${!cuda_device_list[@]}"; do
        cuda_device="${cuda_device_list[$i]}"
        start_index="${start_index_list[$i]}"
        end_index="${end_index_list[$i]}"
        
        image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"
        
        echo "  Starting task ${i}: GPU ${cuda_device}, indices ${start_index}-${end_index}"
        
        # 在后台运行任务
        (
            export CUDA_VISIBLE_DEVICES=${cuda_device}
            python coco_generate_image.py \
                --seed 42 \
                --checkpoint_path "${ckpt_dir}" \
                --model_type sd3 \
                --dataset "${dataset}" \
                --output_dir "${image_dir}" \
                --num_inference_steps 40 \
                --guidance_scale "${cfg_guidance}" \
                --resolution 512 \
                --save_images \
                --mixed_precision fp16 \
                --start_index "${start_index}" \
                --end_index "${end_index}"
        ) &
        
        pids+=($!)

        sleep 30
    done
    
    # 等待所有任务完成
    echo "Waiting for all tasks of method ${method} to complete..."
    for pid in "${pids[@]}"; do
        wait $pid
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Warning: Task with PID $pid exited with code $exit_code"
        fi
    done
    
    echo "=========================================="
    echo "All tasks for method ${method} completed!"
    echo "=========================================="
    echo ""
done

echo "All methods completed!"
