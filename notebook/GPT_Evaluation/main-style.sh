#! /bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate utils

# ==================== 配置多个任务 ====================
# 定义要并行运行的任务列表
# 每个任务是一个数组: (dataset, image_dir_suffix, output_dir_suffix)
# 可以根据需要添加更多任务

tasks=(
    "OneIG-Bench-Portrait:sd-3-5-medium/ckpt-0:sd-3-5-medium"
    "OneIG-Bench-Portrait:FlowGRPO-PickScore/ckpt-0:FlowGRPO-PickScore"
    "OneIG-Bench-Portrait:irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all/ckpt-450:irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all"
)

model="claude-sonnet-4-5-20250929"

# ==================== 并行执行函数 ====================
run_task() {
    local dataset=$1
    local image_dir_suffix=$2
    local output_dir_suffix=$3
    
    local txt_path="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT/dataset/${dataset}/test.txt"
    local image_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/generate_images/${dataset}/${image_dir_suffix}/images"
    local output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/GPT_Evaluation/${dataset}-style_evaluation/${output_dir_suffix}"
    
    echo "=========================================="
    echo "Starting task: ${dataset} - ${output_dir_suffix}"
    echo "Image dir: ${image_dir}"
    echo "Output dir: ${output_dir}"
    echo "=========================================="
    
    python style_evaluation.py \
        --image_dir "${image_dir}" \
        --txt_path "${txt_path}" \
        --model "${model}" \
        --output_dir "${output_dir}"
    
    echo "=========================================="
    echo "Completed task: ${dataset} - ${output_dir_suffix}"
    echo "=========================================="
}

# ==================== 并行执行所有任务 ====================
# 存储所有后台进程的PID
pids=()

# 启动所有任务
for task in "${tasks[@]}"; do
    IFS=':' read -r dataset image_dir_suffix output_dir_suffix <<< "$task"
    run_task "$dataset" "$image_dir_suffix" "$output_dir_suffix" &
    pids+=($!)
    echo "Started task with PID: ${pids[-1]}"
done

# 等待所有任务完成
echo "Waiting for all tasks to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Task with PID $pid completed successfully"
    else
        echo "Task with PID $pid failed with exit code $exit_code"
    fi
done

echo "All tasks completed!"