# `training_sd15/` — RealAlign two-stage training

RealAlign trains in two stages on the (reference, fake) preference pairs
produced by `data_curation/`:

1. **Stage 1 — Diffusion-DRO** (Inverse RL): learns a LoRA against the
   reference image as the expert anchor, producing a warm-start initialization.
2. **Stage 2 — Diffusion-DPO** with LoRA-init: standard Diffusion-DPO over the
   same preference pairs, warm-started from the Stage 1 LoRA.

Stage 2 reads Stage 1's LoRA checkpoint (`pretrained_lora_path` for SD-1.5,
`config.train.lora_path` for SD-3.5-M). The data-curation step
(`data_curation/filter/{hpdv3,external}.py`) writes the CSV both stages
consume as their `csv_file_path` / `train_dataset`.

This folder holds the **SD-1.5** code. SD-3.5-M scripts stay under
`training_sd35m/` because they `import flow_grpo.*` and depend on
`training_sd35m/diffusers_patch/` for the flow-matching SDE samplers —
moving them would require dragging the whole library along.

## Per-model entry points

| Model | Stage 1 (Diffusion-DRO) | Stage 2 (Diffusion-DPO) |
|---|---|---|
| **SD-1.5** | `stage1_diffusion_dro/train-irl.sh` → `train-irl.py` | `stage2_dpo/lora_init.sh` → `train-lora_init.py` |
| **SD-3.5-M** | `../training_sd35m/scripts/single_node/inverse_reinforcement_learning.sh` → `train-sd-3-5-medium-irl.py` | `../training_sd35m/scripts/single_node/dpo.sh` → `train-sd-3-5-medium-dpo.py` |

## Conda env

Both stages share the project env. Every shell script begins with:

```bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop
export HF_ENDPOINT=https://hf-mirror.com   # huggingface mirror (dev cluster)
```

Replace the `conda.sh` path when porting to a new machine.

## Running — SD-1.5

```bash
# Stage 1 (Diffusion-DRO, LoRA)
cd training_sd15/stage1_diffusion_dro
bash train-irl.sh
# → output: <output_dir>/checkpoints/checkpoint-<ckpt>/

# Stage 2 (Diffusion-DPO, LoRA-init from Stage 1)
cd ../stage2_dpo
# edit pretrained_lora_path inside the shell to point at Stage 1's checkpoint
bash lora_init.sh
```

Both shells declare their hyperparameters (`top_N`, `learning_rate`,
`num_steps`, `beta_dpo`, `csv_file_path_train`, …) at the top — edit them in
place rather than passing extra CLI flags.

## Running — SD-3.5-M

```bash
cd training_sd35m

# Stage 1 (Diffusion-DRO / IRL)
bash scripts/single_node/inverse_reinforcement_learning.sh
# → scripts/train-sd-3-5-medium-irl.py
# → config/sd3_5_medium_irl.py:paired_real_fake_dataset_sd3

# Stage 2 (Diffusion-DPO, LoRA-init from Stage 1)
# edit config/sd3_5_medium_dpo.py: train.lora_path to point at Stage 1's checkpoint
bash scripts/single_node/dpo.sh
# → scripts/train-sd-3-5-medium-dpo.py
# → config/sd3_5_medium_dpo.py:paired_real_fake_dataset_sd3
```

Both SD-3.5-M scripts launch via `accelerate launch` on 8 GPUs by default
(`--num_processes=8`); the `:paired_real_fake_dataset_sd3` variant in each
config selects the RealAlign dataset binding. See
`training_sd35m/config/sd3_5_medium_{irl,dpo}.py` for the full schema.

## Layout

```
training_sd15/
├── README.md                      # this file
├── stage1_diffusion_dro/          # SD-1.5 Stage 1 (Diffusion-DRO)
│   ├── train-irl.sh                # canonical launcher
│   ├── train-irl.py          # canonical Stage 1 trainer (LoRA + LoRA init)
│   ├── inference.py / score.py
│   ├── requirements.txt
│   └── tools/ docs/ gradio/ misc/   # upstream-cloned helpers, kept as-is
└── stage2_dpo/                    # SD-1.5 Stage 2 (Diffusion-DPO)
    ├── lora_init.sh               # canonical launcher
    ├── train-lora_init.py         # LoRA-init DPO trainer
    └── LICENSE.txt                # upstream Apache-2.0 (kept for attribution)
```

SD-3.5-M code paths are listed in the entry-points table above; nothing of
SD-3.5-M lives in this directory.
