# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Research workspace for an ICML 2026 submission ("RealAlign") on post-training / preference alignment of text-to-image diffusion models. The repo bundles several upstream research codebases as sibling directories, each with its own training entry point, dependencies, and conventions. There is **no monorepo-level build, setup, or test harness** — work happens inside whichever subproject the task targets.

## Subprojects (top-level layout)

Each of these is effectively an independent project with its own README, scripts, and (usually) `setup.py` / `environment.yaml` / `requirements.txt`. Pick the right one before changing files:

- **`data_curation/`** — RealAlign's dataset pipeline. Builds (reference, fake) preference pairs from HPDv3 (real photos), Pick-a-Pic v2, or Civitai-top. Four stages: `extract/` (uid+prompt CSV) → `construct_pairs/` (U²-Net saliency mask + SD/SD3.5/PixArt inpainting) → `score/` (colorfulness, PickScore, Qwen3-VL anime classifier) → `filter/` (per-source curation; HPDv3 needs `anime drop → color filter → discard negative → top-512`, the others only the last two). Output CSV is consumed by `training_sd15/` and the SD-3.5-M trainers in `flow_grpo_github/`.
- **`training_sd15/`** — RealAlign two-stage SD-1.5 training. `stage1_diffusion_dro/` is Stage 1 (Diffusion-DRO / inverse RL, LoRA + LoRA-init); `stage2_dpo/` is Stage 2 (Diffusion-DPO with LoRA-init warm-started from Stage 1). Each stage has exactly one canonical `train-*.py` + one `*.sh` launcher; argparse-based CLI; launched with `accelerate launch`. SD-3.5-M counterparts live in `flow_grpo_github/scripts/` (see that entry).
- **`flow_grpo_github/`** — Flow-GRPO (online RL for flow-matching models) supporting SD3.5, FLUX.1-dev, FLUX.1-Kontext, Qwen-Image, Qwen-Image-Edit, Wan2.1. Configs use the same `config/grpo.py:VARIANT` / `config/dpo.py:VARIANT` selector pattern. Single-node scripts in `scripts/single_node/`, multi-node in `scripts/multi_node/<model>/`. **Also hosts RealAlign's SD-3.5-M two-stage trainers**: `scripts/train-sd-3-5-medium-irl.py` (Stage 1, launched by `single_node/inverse_reinforcement_learning.sh`) and `scripts/train-sd-3-5-medium-dpo.py` (Stage 2, launched by `single_node/dpo.sh`); both use `config/sd3_5_medium_{irl,dpo}.py:paired_real_fake_dataset_sd3`. They live here (not in `training_sd15/`) because they `import flow_grpo.*` and depend on the local `diffusers_patch/` SDE samplers. RealAlign's SD-3.5-M evaluation lives in `evaluation/sd-3-5-medium/` (`generate_image.py` + `calculate_score.py`); its prompt lists sit alongside Flow-GRPO's own under `dataset/` (`pick_a_pic_v2/`, `partiprompts/`, `drawbench-unique/` — the last is a deduplicated DrawBench, distinct from the existing `drawbench/`). All six metrics (PickScore, ImageReward, Aesthetic, HPSv3, DeQA, UnifiedReward) go through `flow_grpo.rewards.multi_score`.
- **`evaluate_metric/`** — Reward-model and image-quality metrics: `reward_model/{pickscore,hpsv2,aesthetic,clipscore}.py`, plus vendored `clean_fid/`, `cmmd-pytorch/`, `python-cpbd/`, `vila/`. `calculate_metric.py` is the umbrella driver; `evaluate_*.sh` are per-dataset entry points.
- **`benchmark-evaluation/`** — Vendored benchmarks: `geneval/`, `GenEval2/`, `DPG-Bench/`, `OneIG-Benchmark/`, `WISE/`, and the project's own `RealGen/`. Each has its own generation + evaluation scripts, often per base model (`*-sd-v1-5.sh`, `*-sd-3-5-medium.sh`).
- **`notebook/`** — Exploratory Jupyter notebooks (dataset_operation, GPT_Evaluation, ICML-2026, score_analysis, multi_seed_evaluation, …). Not part of any pipeline; treat as scratch.

## How to run things

### Conda environment

Almost every shell script in this repo begins with:
```bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop
export HF_ENDPOINT=https://hf-mirror.com
```
The shared env name is **`alignprop`**. The `HF_ENDPOINT` mirror is set because development happens behind a network where huggingface.co is slow/blocked. When porting a script to a new machine, replace the `conda.sh` path and keep the env activation pattern.

flow_grpo_github also accepts a fresh env created from its own `setup.py` (`pip install -e .`), pinned to `torch==2.6.0`, `diffusers==0.33.1`, `transformers==4.40.0`, `accelerate==1.4.0`, Python 3.10.

### Launchers by subproject

The launcher tool differs by subproject — do not mix them:

| Subproject | Launcher | Config style |
|---|---|---|
| `training_sd15/stage1_diffusion_dro/` | `accelerate launch --multi_gpu --num_processes=N train-irl.py --flag …` (via `train-irl.sh`) | argparse CLI |
| `training_sd15/stage2_dpo/` | `accelerate launch --mixed_precision=fp16 train-lora_init.py --flag …` (via `lora_init.sh`) | argparse CLI |
| `flow_grpo_github/` | `accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml scripts/train_*.py --config config/grpo.py:VARIANT` | same `:variant` selector pattern |

The `:VARIANT` config selector (flow_grpo_github) is a custom convention where the name after the colon picks a top-level function in the config module (e.g. `sd3_geneval`, `general_ocr_wan2_1`, `pickscore_sd3_fast_nocfg`). Look in the config file to enumerate valid variants before guessing.

### Per-subproject bash entry scripts

Most subprojects keep one or more `script_command*.sh` / `lora_init*.sh` / `*-script-*.sh` files at their root that wrap the launcher with the actual flags used in experiments. These are the most reliable source of "the exact command we run" — when reproducing an experiment, start from the existing shell script, do not hand-construct the `accelerate launch` line.

## Cross-cutting notes

- **Hardcoded absolute paths.** Many scripts and configs contain paths under `/data_center/data2/dataset/chenwy/21164-data/...` (datasets, model checkpoints, AIGI detectors), `/data3/chenweiyan/...`, or specific HuggingFace cache dirs. When reading or modifying scripts, expect these and do not assume they exist locally; flag them when they appear in code you are asked to run.
- **`flow_grpo_github/diffusers_patch/`** overrides pieces of `diffusers` (notably `pipeline_with_logprob.py`, SD3/Flux SDE samplers, `train_dreambooth_lora_sd3.py`). When debugging sampling/log-prob issues, inspect the patch before assuming upstream `diffusers` behavior.
- **Multi-reward training (Flow-GRPO)** takes a dict like `{"pickscore": 0.5, "ocr": 0.2, "aesthetic": 0.3}`. Supported reward names are defined in `flow_grpo_github/flow_grpo/rewards.py`; some (GenEval) require running a separate reward server and may need their own conda env. DeQA / UnifiedReward / HPSv3 are loaded locally and used by the evaluation harness in `flow_grpo_github/evaluation/`.
- **fp16 vs bf16 for RL training.** Per the Flow-GRPO README, fp16 is preferred (smaller log-prob error between sampling and training) — but Flux and Wan must use bf16 because fp16 inference produces broken outputs. SD3/SDXL/SD1.5 use fp16.
- **`.gitignore`** excludes vendored evaluation deps (`evaluate_metric/hf_cache`, `evaluate_metric/t2v_metrics_github`, `evaluate_metric/MA-AGIQA`, `evaluate_metric/PKU-AIGIQA-4K`, `evaluate_metric/General-Visual-Quality-RL`) and `benchmark-evaluation/RealGen/benchmark/real-img-benchmark` — these directories may be referenced by scripts but are not checked in.

## Editing guidelines specific to this repo

- For RealAlign's two-stage training there is exactly one canonical trainer per (stage × model): `training_sd15/stage{1_diffusion_dro,2_dpo}/train-*.py` for SD-1.5 and `flow_grpo_github/scripts/train-sd-3-5-medium-{irl,dpo}.py` for SD-3.5-M. Historical ablation variants (`-w_sft`, `-curriculum_learning`, `-l_diff_coeffi`, `_maskdpo`, `-dpo_next`, `-w_sft_dgr`, `-only_sft`, `-iterative_dpo_sft`, `-dpo-sft`) have been removed; do not reintroduce sibling scripts. Edit the relevant `*.sh` launcher's hyperparameters in place rather than forking the trainer.
- The `notebook/` folder and `extract_prompts.py` at the repo root are throwaway helpers — do not refactor them unless asked.
