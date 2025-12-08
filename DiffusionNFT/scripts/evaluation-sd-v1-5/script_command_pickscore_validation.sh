#!/bin/bash

cuda_devices=(${1:-"0 1 2 3"})
method=${2:-"random_9984_images_no_anime_pickscore_002-hpdv3_all-inpainting"}
ckpt_list=(${3:-"500 600 700 800"})

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="script_command.sh"

if [ ! -f "${SCRIPT_DIR}/${SCRIPT_NAME}" ]; then
    echo "Error: ${SCRIPT_NAME} not found in ${SCRIPT_DIR}"
    exit 1
fi

if [ ${#cuda_devices[@]} -lt ${#ckpt_list[@]} ]; then
    echo "Warning: GPU 数量 (${#cuda_devices[@]}) 少于 ckpt 数量 (${#ckpt_list[@]})，部分任务将等待 GPU 资源"
fi

echo "=========================================="
echo "执行配置:"
echo "  CUDA Devices: ${cuda_devices[@]} (共 ${#cuda_devices[@]} 个)"
echo "  Method: ${method}"
echo "  Checkpoints: ${ckpt_list[@]} (共 ${#ckpt_list[@]} 个)"
echo "=========================================="

# 创建临时目录存储日志
LOG_DIR="${SCRIPT_DIR}/logs_${method}"
mkdir -p "${LOG_DIR}"
echo "日志目录: ${LOG_DIR}"
echo ""

pids=()
for i in "${!ckpt_list[@]}"; do
    ckpt=${ckpt_list[$i]}
    gpu_idx=$((i % ${#cuda_devices[@]}))
    cuda_device=${cuda_devices[$gpu_idx]}
    
    log_file="${LOG_DIR}/ckpt_${ckpt}_gpu_${cuda_device}.log"
    echo "启动任务: ckpt=${ckpt}, GPU=${cuda_device}, 日志=${log_file}"
    bash "${SCRIPT_DIR}/${SCRIPT_NAME}" "${cuda_device}" "${method}" "${ckpt}" > "${log_file}" 2>&1 &
    pids+=($!)
    echo "  PID: ${pids[-1]}"
done

echo ""
echo "已启动 ${#pids[@]} 个任务，等待所有任务完成..."
echo ""

# 等待所有任务完成
success_count=0
fail_count=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    ckpt=${ckpt_list[$i]}
    gpu_idx=$((i % ${#cuda_devices[@]}))
    cuda_device=${cuda_devices[$gpu_idx]}
    
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ ckpt ${ckpt} (GPU ${cuda_device}) 完成 (PID: ${pid})"
        success_count=$((success_count + 1))
    else
        echo "✗ ckpt ${ckpt} (GPU ${cuda_device}) 失败 (PID: ${pid}, exit code: ${exit_code})"
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "=========================================="
echo "任务完成统计:"
echo "  成功: ${success_count}"
echo "  失败: ${fail_count}"
echo "  总计: ${#pids[@]}"
echo "  日志目录: ${LOG_DIR}"
echo "=========================================="

echo ""
echo "=========================================="
echo "所有任务执行完成"
echo "=========================================="

