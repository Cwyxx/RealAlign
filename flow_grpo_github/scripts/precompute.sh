#!/bin/bash

CONFIG="config/sd3_5_medium_dpo.py:paired_real_fake_dataset_sd3"
SCRIPT_PATH="scripts/precompute_prompt_embeddings.py"
CHUNK_SIZE=20000
GPU_IDS=(0 1 2 3)  # 使用的显卡 ID 列表
for i in $(seq 0 3)
do
    START=$((i * CHUNK_SIZE + 600000))
    END=$(((i + 1) * CHUNK_SIZE + 600000))
    GPU_ID=${GPU_IDS[$i]}  # 从数组中获取对应的 GPU ID
    
    echo "正在显卡 $GPU_ID 上启动进程：区间 [$START, $END)"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT_PATH \
        --config $CONFIG \
        --start_index $START \
        --end_index $END & 
    sleep 60
done

# 等待所有后台任务完成
wait
echo "所有数据预处理任务已完成！"