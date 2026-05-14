# `training_sd15/` — RealAlign SD-1.5 two-stage training

The **SD-1.5** implementation of RealAlign's two-stage training pipeline (Stage 1 Diffusion-DRO / inverse RL → Stage 2 Diffusion-DPO with LoRA-init).

## 1️⃣ Stage 1 — Diffusion-DRO / Inverse RL

### 📥 Input format

Stage 1 reads an **expert-image dataset directory**. Each sample is a subdirectory named by `uid`, containing a prompt file and a preferred/reference image:

```text
expert_image_dataset/
├── <uid_0>/
│   ├── caption.txt              # prompt text
│   └── <uid_0>.png              # expert/reference image
├── <uid_1>/
│   ├── caption.txt
│   └── <uid_1>.png
└── ...
```

Convert the final CSV produced by [`../data_curation/`](../data_curation/) into this layout (run from the RealAlign repository root):

```bash
python training_sd15/export_stage1_irl.py \
  --csv-file /path/to/final_training.csv \
  --output-dir /path/to/stage1_expert_image_dataset
```

The exporter copies `real_image_path` and writes `prompt` into `caption.txt`. If the CSV only carries `uid` and `prompt`, also pass `--image-dir /path/to/reference_images`; images are then resolved as `<image-dir>/<uid>.png`.

### 🚀 Run

```bash
cd training_sd15/stage1_diffusion_dro
bash train-irl.sh
```

The launcher runs:

```bash
accelerate launch --multi_gpu --num_processes 4 train-irl.py ...
```

Edit the variables at the top of [`stage1_diffusion_dro/train-irl.sh`](stage1_diffusion_dro/train-irl.sh) before launching:

- `train_dataset` — expert-image dataset directory produced above.
- `validation_dataset` — validation prompt/image directory.
- `unet_init` — base SD-1.5 model, usually `runwayml/stable-diffusion-v1-5`.
- `output_dir` / `run_name` — where checkpoints and logs are written.
- `num_steps`, `learning_rate`, `top_N`, `MASTER_PORT`, `CUDA_VISIBLE_DEVICES`.

### 💾 Output

Checkpoints are saved under:

```text
<output_dir>/checkpoints/checkpoint-<step>/
```

These serve as the `pretrained_lora_path` for Stage 2.

## 2️⃣ Stage 2 — Diffusion-DPO with LoRA-init

Stage 2 warm-starts from a Stage 1 LoRA checkpoint via `--pretrained_lora_path` and continues training with Diffusion-DPO on paired `(real, fake)` data.

### 📥 Input format

Stage 2 reads the final CSV produced by [`../data_curation/`](../data_curation/) directly — no expert-image directory conversion needed:

```text
final_training.csv
├── uid                  # unique pair id
├── prompt               # prompt shared by the preferred and dispreferred images
├── real_image_path      # preferred image; the real / high-quality reference
└── fake_image_path      # dispreferred image; the fake generated counterpart
```

Extra curation metadata columns (PickScore, colorfulness, ...) may remain in the CSV; the Stage 2 dataloader ignores them.

### 🚀 Run

Before launching, set `pretrained_lora_path` in [`stage2_dpo/lora_init.sh`](stage2_dpo/lora_init.sh) to the Stage 1 checkpoint you want to warm-start from.

```bash
cd training_sd15/stage2_dpo
bash lora_init.sh
```

The launcher runs:

```bash
accelerate launch --mixed_precision="fp16" train-lora_init.py ...
```

Edit the variables at the top of [`stage2_dpo/lora_init.sh`](stage2_dpo/lora_init.sh) before launching:

- `csv_file_path_train` — paired `(real, fake)` training CSV from data curation.
- `pretrained_lora_path` — Stage 1 LoRA checkpoint.
- `unet_init` — base SD-1.5 model, usually `runwayml/stable-diffusion-v1-5`.
- `output_dir` / `run_name` — where checkpoints and logs are written.
- `beta_dpo`, `top_N`, `ckpt`, `CUDA_VISIBLE_DEVICES`.

## 📁 Layout

```text
training_sd15/
├── README.md
├── export_stage1_irl.py
├── stage1_diffusion_dro/
│   ├── train-irl.sh
│   ├── train-irl.py
│   ├── inference.py
│   ├── score.py
│   ├── requirements.txt
│   └── tools/ docs/ gradio/ misc/
└── stage2_dpo/
    ├── lora_init.sh
    ├── train-lora_init.py
    └── LICENSE.txt
```
