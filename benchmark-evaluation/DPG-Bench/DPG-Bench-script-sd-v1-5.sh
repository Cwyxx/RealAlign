#! /bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

# Usage:
#   bash DPG-Bench-script-sd-v1-5.sh <cuda_device> <method> <ckpt_step> <rl_framework>
#
# Example:
#   bash DPG-Bench-script-sd-v1-5.sh 0 sd-v1-5 5000 diffusion-dpo
#
# Arguments:
#   cuda_device   — visible CUDA device index (e.g. 0)
#   method        — model/run name used to locate the checkpoint directory
#   ckpt_step     — checkpoint step number (e.g. 5000)
#   rl_framework  — RL framework name used in checkpoint/image paths

cuda_device=$1   # e.g. 0
method=$2        # e.g. "sd-v1-5"
ckpt=$3          # e.g. 5000
rl_framework=$4  # e.g. "diffusion-dpo"

export CUDA_VISIBLE_DEVICES=${cuda_device}
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

# --- Paths ---
dpg_bench_dir="$(cd "$(dirname "$0")" && pwd)"
prompts_dir="${dpg_bench_dir}/dpg_bench/prompts"
csv_path="${dpg_bench_dir}/dpg_bench/dpg_bench.csv"

base_ckpt_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/model-ckpt"
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"

base_image_dir="/data_center/data2/dataset/chenwy/21164-data/${rl_framework}/sd-v1-5/generate_images/dpg-bench"
image_dir="${base_image_dir}/${method}/ckpt-${ckpt}"

unet_init="runwayml/stable-diffusion-v1-5"
if [[ "$method" == *"dpo-official"* ]]; then
    unet_init="mhdang/dpo-sd1.5-text2image-v1"
fi

resolution=512
pic_num=4     # 4 = generate 2×2 grid per prompt (matches dist_eval.sh default)
num_gpus=1    # number of GPUs for evaluation (accelerate)
eval_port=29500

echo "============================================================"
echo " DPG-Bench evaluation: SD v1.5"
echo "  Method       : ${method}"
echo "  Checkpoint   : ${ckpt_dir}"
echo "  Output dir   : ${image_dir}"
echo "  Resolution   : ${resolution}  |  pic_num: ${pic_num}"
echo "============================================================"

# # ============================================================
# # Step 1 – Generate images
# # ============================================================
# echo ""
# echo "[1/2] Generating images..."
# python "${dpg_bench_dir}/generate-image-sd-v1-5.py" \
#     --prompts_dir "${prompts_dir}" \
#     --checkpoint_path "${ckpt_dir}" \
#     --output_dir "${image_dir}" \
#     --unet_init "${unet_init}" \
#     --resolution ${resolution} \
#     --pic_num ${pic_num} \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --seed 42 \
#     --skip_existing

# ============================================================
# Step 2 – Evaluate with DPG-Bench (mplug VQA)
# ============================================================
echo ""
echo "[2/2] Running DPG-Bench evaluation..."
conda activate geneval2   # switch to the environment that has modelscope/mplug deps

cd "${dpg_bench_dir}"

PIC_NUM=${pic_num} PROCESSES=${num_gpus} PORT=${eval_port} \
    bash dpg_bench/dist_eval.sh "${image_dir}" ${resolution}

echo ""
echo "Done. Results saved alongside images in: ${image_dir}"
