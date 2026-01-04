#! /bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate utils

# ================= Configuration =================
dataset="HPDv2-anime"
txt_path="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT/dataset/${dataset}/test.txt"
image1_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/generate_images_seed_42/${dataset}/irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all/ckpt-450/images"
model="gemini-3-flash-preview" # "grok-4-1-fast-reasoning" # "gemini-3-flash-preview" # "claude-haiku-4-5-20251001" # "gpt-5.1-chat-latest" "grok-4-1-fast-non-reasoning"
echo "image1_dir: ${image1_dir}"
echo "model: ${model}"

# 支持多个 vs_method，用空格分隔
vs_methods=("DiffusionNFT" "FlowGRPO-PickScore" "GRPO-Guard")  # 在这里添加你需要的所有方法

# 并行执行数量（同时运行的任务数），可以根据系统资源调整
max_parallel=5

# ================= Functions =================
run_evaluation() {
    local vs_method=$1
    local image2_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/generate_images_seed_42/${dataset}/${vs_method}/ckpt-0/images"
    local output_dir="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/GPT_Evaluation_v3/${dataset}/hpdv3_all_ckpt_450-vs-${vs_method}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation for vs_method: ${vs_method}"
    
    python main.py \
        --txt_path ${txt_path} \
        --model ${model} \
        --image1_dir ${image1_dir} \
        --image2_dir ${image2_dir} \
        --output_dir ${output_dir}
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed evaluation for vs_method: ${vs_method}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed evaluation for vs_method: ${vs_method} (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# ================= Main Execution =================
echo "=========================================="
echo "Starting GPT Evaluation for ${#vs_methods[@]} method(s)"
echo "Dataset: ${dataset}"
echo "Model: ${model}"
echo "Max parallel: ${max_parallel}"
echo "Methods: ${vs_methods[@]}"
echo "=========================================="

# 使用数组来跟踪后台进程
pids=()
running=0
index=0

# 循环处理所有 vs_methods
while [ $index -lt ${#vs_methods[@]} ] || [ $running -gt 0 ]; do
    # 启动新任务（如果还有未处理的方法且未达到最大并行数）
    while [ $running -lt $max_parallel ] && [ $index -lt ${#vs_methods[@]} ]; do
        vs_method=${vs_methods[$index]}
        run_evaluation "$vs_method" &
        pid=$!
        pids+=($pid)
        echo "[PID: $pid] Started background task for: ${vs_method}"
        ((index++))
        ((running++))
    done
    
    # 等待至少一个进程完成
    if [ $running -gt 0 ]; then
        wait -n  # 等待任意一个后台进程完成
        exit_code=$?
        ((running--))
        
        # 从 pids 数组中移除已完成的进程（简化处理）
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 $pid 2>/dev/null; then
                new_pids+=($pid)
            fi
        done
        pids=("${new_pids[@]}")
    fi
done

# 等待所有剩余的后台进程完成
echo "Waiting for all remaining tasks to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Warning: Process $pid exited with code $exit_code"
    fi
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="