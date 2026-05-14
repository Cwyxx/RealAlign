# `training_sd35m/` — RealAlign SD-3.5-M training

The **SD-3.5-M** implementation of RealAlign's two-stage training pipeline, built on the local Flow-GRPO codebase (many upstream Flow-GRPO utilities, scripts, and `diffusers_patch/` overrides remain in this directory). For SD-1.5, see [`../training_sd15/`](../training_sd15/).

## 📥 Input data

Both stages share the same input — paired `(real, fake)` CSV files from [`../data_curation/`](../data_curation/):

```text
final_training.csv
├── uid                  # unique pair id; also used to find precomputed embeddings
├── prompt               # text prompt shared by the preferred and dispreferred images
├── real_image_path      # preferred image; the real / high-quality reference
└── fake_image_path      # dispreferred image; the generated fake counterpart
```

The trainers also load **precomputed SD-3.5 prompt embeddings** keyed by `uid`. Precomputing the embeddings once — rather than re-encoding prompts every step — lets training drop the SD-3.5 text encoders from VRAM, which substantially cuts memory cost. Each CSV row needs matching `{uid}.pt` and `{uid}_pooled.pt` files under the directory configured in:

- `config.irl.precomputed_embeddings_dir_dict` — Stage 1
- `config.dpo.precomputed_embeddings_dir_dict` — Stage 2

If a CSV does not yet have these embedding files, generate them with [`scripts/precompute_prompt_embeddings.py`](scripts/precompute_prompt_embeddings.py) (or the `_with_range.py` variant for chunked precomputation).

## 1️⃣ Stage 1 — Diffusion-DRO / Inverse RL

### 🚀 Run

```bash
cd training_sd35m
bash scripts/single_node/inverse_reinforcement_learning.sh
```

The launcher runs:

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=8 \
  --main_process_port 29501 \
  scripts/train-sd-3-5-medium-irl.py \
  --config config/sd3_5_medium_irl.py:paired_real_fake_dataset_sd3
```

The `:paired_real_fake_dataset_sd3` suffix selects a top-level function inside the config file. Edit that function in [`config/sd3_5_medium_irl.py`](config/sd3_5_medium_irl.py) before launching:

- `config.irl.csv_file_path` — training and validation CSVs.
- `config.irl.precomputed_embeddings_dir_dict` — matching prompt-embedding directories.
- `config.irl.dataset` — which CSV keys to use for train and validation.
- `config.train.lora_path` — optional initial LoRA, usually a previous SD-3.5-M alignment checkpoint.
- `config.run_name` / `config.save_dir` — output naming and checkpoint location.
- `config.irl.max_train_steps`, `config.train.learning_rate`, `config.train.beta`, batch settings.

### 💾 Output

Checkpoints are saved under:

```text
<config.save_dir>/checkpoints/checkpoint-<step>/lora/learner
```

This path is what Stage 2's `config.train.lora_path` should point at.

## 2️⃣ Stage 2 — Diffusion-DPO with LoRA-init

Stage 2 warm-starts from a Stage 1 LoRA checkpoint via `config.train.lora_path` and continues training with Diffusion-DPO on the same paired `(real, fake)` data.

### 🚀 Run

Before launching, set `config.train.lora_path` in [`config/sd3_5_medium_dpo.py`](config/sd3_5_medium_dpo.py) to the Stage 1 LoRA checkpoint you want to warm-start from.

```bash
cd training_sd35m
bash scripts/single_node/dpo.sh
```

The launcher runs:

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=8 \
  --main_process_port 29501 \
  scripts/train-sd-3-5-medium-dpo.py \
  --config config/sd3_5_medium_dpo.py:paired_real_fake_dataset_sd3
```

Edit the `:paired_real_fake_dataset_sd3` function in [`config/sd3_5_medium_dpo.py`](config/sd3_5_medium_dpo.py) before launching:

- `config.dpo.csv_file_path` — paired training and validation CSVs.
- `config.dpo.precomputed_embeddings_dir_dict` — matching prompt-embedding directories.
- `config.dpo.dataset` — which CSV keys to use for train and validation.
- `config.train.lora_path` — Stage 1 LoRA checkpoint for LoRA-init.
- `config.run_name` / `config.save_dir` — output naming and checkpoint location.
- `config.dpo.max_train_steps`, `config.train.learning_rate`, `config.train.beta`, batch settings.

## 📊 Evaluation

RealAlign evaluation scripts for SD-3.5-M live in [`evaluation/sd-3-5-medium/`](evaluation/sd-3-5-medium/):

- [`generate_image.py`](evaluation/sd-3-5-medium/generate_image.py) — generates images from a model or LoRA checkpoint.
- [`calculate_score.py`](evaluation/sd-3-5-medium/calculate_score.py) — computes reward-model scores.
- [`run_multi_seed_eval.sh`](evaluation/sd-3-5-medium/run_multi_seed_eval.sh) — wraps multi-seed evaluation.

Prompt lists used by the evaluation harness live under [`dataset/`](dataset/), including `pick_a_pic_v2/`, `partiprompts/`, and `drawbench-unique/`. The six supported reward metrics — PickScore, ImageReward, Aesthetic, HPSv3, DeQA, UnifiedReward — are routed through `flow_grpo.rewards.multi_score`.

## 📁 Layout

```text
training_sd35m/
├── README.md
├── export_stage1_irl.py -> ../training_sd15/export_stage1_irl.py
├── config/
│   ├── sd3_5_medium_irl.py
│   ├── sd3_5_medium_dpo.py
│   └── base.py
├── scripts/
│   ├── single_node/
│   │   ├── inverse_reinforcement_learning.sh
│   │   └── dpo.sh
│   ├── train-sd-3-5-medium-irl.py
│   ├── train-sd-3-5-medium-dpo.py
│   └── precompute_prompt_embeddings*.py
├── diffusers_patch/
├── flow_grpo/
├── evaluation/
│   ├── sd-3-5-medium/
│   └── sd-v1-5/
└── dataset/
```
