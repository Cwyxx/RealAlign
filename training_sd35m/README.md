# `training_sd35m/` - RealAlign SD-3.5-M training

This folder contains the **SD-3.5-M** implementation of RealAlign's two-stage training pipeline. It is built on the local Flow-GRPO codebase, so many
upstream Flow-GRPO utilities, scripts, and patches remain in this directory, but the RealAlign SD-3.5-M entry points are the two trainers listed below.

For the SD-1.5 trainers, see [`../training_sd15/`](../training_sd15/).

## RealAlign entry points

| Stage | Launcher | Trainer | Config |
|---|---|---|---|
| Stage 1 | [`scripts/single_node/inverse_reinforcement_learning.sh`](scripts/single_node/inverse_reinforcement_learning.sh) | [`scripts/train-sd-3-5-medium-irl.py`](scripts/train-sd-3-5-medium-irl.py) | [`config/sd3_5_medium_irl.py:paired_real_fake_dataset_sd3`](config/sd3_5_medium_irl.py) |
| Stage 2 | [`scripts/single_node/dpo.sh`](scripts/single_node/dpo.sh) | [`scripts/train-sd-3-5-medium-dpo.py`](scripts/train-sd-3-5-medium-dpo.py) | [`config/sd3_5_medium_dpo.py:paired_real_fake_dataset_sd3`](config/sd3_5_medium_dpo.py) |

The `:paired_real_fake_dataset_sd3` suffix selects a top-level function inside the config file. Check that function before launching whenever you change datasets, LoRA initialization, output paths, or training length.

## Environment

The experiment launchers expect the shared `alignprop` conda environment and begin with:

```bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop
export HF_ENDPOINT=https://hf-mirror.com
```

The pinned versions live in [`setup.py`](setup.py), including `torch==2.6.0`, `diffusers==0.33.1`, `transformers==4.40.0`, `accelerate==1.4.0`, and Python 3.10+.

## Dataset inputs

Both SD-3.5-M stages use paired `(real, fake)` CSV files. Each row is expected
to include at least:

- `uid`
- `prompt`
- `win_image_path`
- `lose_image_path`

The trainers also load precomputed SD-3.5 prompt embeddings from the directories
configured in:

- `config.irl.precomputed_embeddings_dir_dict` for Stage 1.
- `config.dpo.precomputed_embeddings_dir_dict` for Stage 2.

Use [`scripts/precompute_prompt_embeddings.py`](scripts/precompute_prompt_embeddings.py) or [`scripts/precompute_prompt_embeddings_with_range.py`](scripts/precompute_prompt_embeddings_with_range.py) when a new CSV does not already have matching `{uid}.pt` and `{uid}_pooled.pt` embedding files.

## Run Stage 1

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

Before launching, edit
[`config/sd3_5_medium_irl.py`](config/sd3_5_medium_irl.py):

- `config.irl.csv_file_path`: training and validation CSVs.
- `config.irl.precomputed_embeddings_dir_dict`: matching prompt-embedding directories.
- `config.irl.dataset`: which CSV keys to use for train and validation.
- `config.train.lora_path`: optional initial LoRA, usually a previous SD-3.5-M alignment checkpoint.
- `config.run_name` and `config.save_dir`: output naming and checkpoint location.
- `config.irl.max_train_steps`, `config.train.learning_rate`, `config.train.beta`, and batch settings.

## Run Stage 2

Before Stage 2, set `config.train.lora_path` in [`config/sd3_5_medium_dpo.py`](config/sd3_5_medium_dpo.py) to the Stage 1 LoRA checkpoint you want to warm-start from. Stage 1 checkpoints are saved under a path like:

```text
<stage1-save-dir>/checkpoints/checkpoint-<step>/lora/learner
```

Then launch:

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

Before launching, edit
[`config/sd3_5_medium_dpo.py`](config/sd3_5_medium_dpo.py):

- `config.dpo.csv_file_path`: paired training and validation CSVs.
- `config.dpo.precomputed_embeddings_dir_dict`: matching prompt-embedding directories.
- `config.dpo.dataset`: which CSV keys to use for train and validation.
- `config.train.lora_path`: Stage 1 LoRA checkpoint for LoRA-init.
- `config.run_name` and `config.save_dir`: output naming and checkpoint location.
- `config.dpo.max_train_steps`, `config.train.learning_rate`, `config.train.beta`, and batch settings.

## Evaluation

RealAlign evaluation scripts for SD-3.5-M live in
[`evaluation/sd-3-5-medium/`](evaluation/sd-3-5-medium/):

- [`generate_image.py`](evaluation/sd-3-5-medium/generate_image.py) generates images from a model or LoRA checkpoint.
- [`calculate_score.py`](evaluation/sd-3-5-medium/calculate_score.py) computes reward-model scores.
- [`run_multi_seed_eval.sh`](evaluation/sd-3-5-medium/run_multi_seed_eval.sh) wraps multi-seed evaluation.

Prompt lists used by the evaluation harness live under [`dataset/`](dataset/), including `pick_a_pic_v2/`, `partiprompts/`, and `drawbench-unique/`.
The six supported reward metrics, PickScore, ImageReward, Aesthetic, HPSv3, DeQA, and UnifiedReward, are routed through `flow_grpo.rewards.multi_score`.

## Layout

```text
training_sd35m/
├── README.md
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
