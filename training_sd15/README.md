# `training_sd15/` — RealAlign SD-1.5 training

This folder contains the **SD-1.5** implementation of RealAlign's two-stage training pipeline. It uses the preference data produced by `../data_curation/`, but the two stages expect different input formats:

- **Stage 1 — Diffusion-DRO / inverse RL** reads an expert-image dataset directory through `--train_dataset`. Each sample directory contains a
  `caption.txt` file and a `.png` image.
- **Stage 2 — Diffusion-DPO with LoRA-init** reads the paired `(real, fake)` CSV through `--csv_file_path_train` and warm-starts from the Stage 1 LoRA checkpoint through `--pretrained_lora_path`.

SD-3.5-M training is maintained separately under `../training_sd35m/`; this README only documents the SD-1.5 code in this directory.

## Entry points

| Stage | Launcher | Trainer | Main inputs |
|---|---|---|---|
| Stage 1 | [`stage1_diffusion_dro/train-irl.sh`](stage1_diffusion_dro/train-irl.sh) | [`stage1_diffusion_dro/train-irl.py`](stage1_diffusion_dro/train-irl.py) | `--train_dataset`, `--validation_dataset`, `--unet_init` |
| Stage 2 | [`stage2_dpo/lora_init.sh`](stage2_dpo/lora_init.sh) | [`stage2_dpo/train-lora_init.py`](stage2_dpo/train-lora_init.py) | `--csv_file_path_train`, `--pretrained_lora_path`, `--unet_init` |

## Environment

Both launchers expect the shared `alignprop` conda environment and begin with:

```bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop
export HF_ENDPOINT=https://hf-mirror.com
```

Replace the `conda.sh` path when porting to another machine. The launchers also set `CUDA_VISIBLE_DEVICES`; edit that variable before running if your GPU layout differs.

## Run Stage 1

```bash
cd training_sd15/stage1_diffusion_dro
bash train-irl.sh
```

The launcher runs:

```bash
accelerate launch --multi_gpu --num_processes 4 train-irl.py ...
```

Edit the variables at the top of [`stage1_diffusion_dro/train-irl.sh`](stage1_diffusion_dro/train-irl.sh) before launching:

- `train_dataset`: expert-image dataset directory used by Stage 1.
- `validation_dataset`: validation prompt/image directory.
- `unet_init`: base SD-1.5 model, usually `runwayml/stable-diffusion-v1-5`.
- `output_dir` / `run_name`: where checkpoints and logs are written.
- `num_steps`, `learning_rate`, `top_N`, `MASTER_PORT`, `CUDA_VISIBLE_DEVICES`.

Stage 1 checkpoints are saved under:

```text
<output_dir>/checkpoints/checkpoint-<step>/
```

## Run Stage 2

Before launching Stage 2, set `pretrained_lora_path` in [`stage2_dpo/lora_init.sh`](stage2_dpo/lora_init.sh) to the Stage 1 checkpoint you want to use.

```bash
cd training_sd15/stage2_dpo
bash lora_init.sh
```

The launcher runs:

```bash
accelerate launch --mixed_precision="fp16" train-lora_init.py ...
```

Edit the variables at the top of
[`stage2_dpo/lora_init.sh`](stage2_dpo/lora_init.sh) before launching:

- `csv_file_path_train`: paired `(real, fake)` training CSV from data curation.
- `pretrained_lora_path`: Stage 1 LoRA checkpoint.
- `unet_init`: base SD-1.5 model, usually `runwayml/stable-diffusion-v1-5`.
- `output_dir` / `run_name`: where checkpoints and logs are written.
- `beta_dpo`, `top_N`, `ckpt`, `CUDA_VISIBLE_DEVICES`.

## Layout

```text
training_sd15/
├── README.md
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
