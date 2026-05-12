<h1 align="center">When Preference Labels Fall Short:<br>Aligning Diffusion Models from Real Data</h1>

<div align="center">
  <a href='TODO'><img src='https://img.shields.io/badge/OpenReview-red?logo=openreview'></a> &nbsp;
  <a href='TODO'><img src='https://img.shields.io/badge/Project_Page-9E95B7?logo=github'></a> &nbsp;
  <a href='TODO'><img src='https://img.shields.io/badge/Code-9E95B7?logo=github'></a> &nbsp;
  <a href='TODO'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp;
  <a href='TODO'><img src='https://img.shields.io/badge/Dataset-blue?logo=huggingface'></a>
</div>

<p align="center"><em>Accepted at ICML 2026. Links above are placeholders — TODO before public release.</em></p>

## 📝 Abstract

Preference alignment aims to guide generative models by learning from comparisons between preferred and non-preferred samples. In practice, most existing approaches rely on preference pairs constructed from model-generated images. Such supervision is inherently relative and can be ambiguous when both samples exhibit artifacts or limited visual quality, making it difficult to infer what constitutes a truly desirable output. In this work, we investigate whether real data can serve as an alternative source of supervision for preference alignment. We adopt a data-centric perspective and study a curation strategy that treats real images as reference points and constructs preference signals by contrasting them with generated or perturbed samples, without requiring manually annotated preference pairs. Through empirical analysis, we show that real-data-based supervision provides effective guidance for aligning diffusion models and achieves performance comparable to existing preference-based methods. Our results suggest that real data offers a practical and complementary source of supervision for preference alignment and highlight directions of label-efficient alignment strategies.

## 🖼️ Gallery

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="assets/gallery/00009.png" alt="gallery 00009" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00368.png" alt="gallery 00368" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00149.png" alt="gallery 00149" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00150.png" alt="gallery 00150" width="100%"/></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="assets/gallery/00243.png" alt="gallery 00243" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/01175.png" alt="gallery 01175" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00269.png" alt="gallery 00269" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00264.png" alt="gallery 00264" width="100%"/></td>
  </tr>
</table>

## 🗂️ Repository layout

Only the directories below are part of RealAlign itself. Each top-level folder is an independent subproject with its own scripts and dependencies.

**Data curation**

- [`data_curation/`](data_curation/) builds `(real, fake)` preference pairs from HPDv3, Pick-a-Pic v2, and Civitai-top.
- The pipeline follows four stages: `extract/` → `construct_pairs/` → `score/` → `filter/`.
- The curated CSV is consumed by both SD-1.5 and SD-3.5-M training.

**Training**

- [`training_sd15/`](training_sd15/) contains the RealAlign SD-1.5 two-stage trainers:
  - `stage1_diffusion_dro/train-irl.py` for Stage 1 Diffusion-DRO / inverse RL.
  - `stage2_dpo/train-lora_init.py` for Stage 2 Diffusion-DPO, warm-started from the Stage 1 LoRA.
- [`training_sd35m/scripts/`](training_sd35m/scripts/) contains the RealAlign SD-3.5-M two-stage trainers:
  - `train-sd-3-5-medium-irl.py` for Stage 1.
  - `train-sd-3-5-medium-dpo.py` for Stage 2.
  - These live inside `training_sd35m/` because they depend on `flow_grpo.*` and the local `diffusers_patch/` SDE samplers.

**Evaluation**

- [`training_sd35m/evaluation/`](training_sd35m/evaluation/) contains the reward-model evaluation harness for SD-3.5-M.
- [`training_sd35m/evaluation/sd-3-5-medium/`](training_sd35m/evaluation/sd-3-5-medium/) provides `generate_image.py` and `calculate_score.py`.
- Prompt lists live under `training_sd35m/dataset/`, including `pick_a_pic_v2/`, `partiprompts/`, and `drawbench-unique/`.
- All six reward metrics, PickScore, ImageReward, Aesthetic, HPSv3, DeQA, and UnifiedReward, are routed through `flow_grpo.rewards.multi_score`.
- [`DPG-Bench/`](DPG-Bench/) contains DPG-Bench evaluation scripts for SD-1.5 and SD-3.5-M.

**Paper artifacts**

- [`notebook/`](notebook/) contains curated ICML 2026 analysis notebooks and figure artifacts used in the paper.

## 🚀 Quick start

### 1. Environment

Every shell script in this repo expects a single `alignprop` conda env and begins with:

```bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop
export HF_ENDPOINT=https://hf-mirror.com   # huggingface mirror (dev cluster)
```

Replace the `conda.sh` path when porting to a new machine. The env pins `torch==2.6.0`, `diffusers==0.33.1`, `transformers==4.40.0`, `accelerate==1.4.0`, Python 3.10. See [`training_sd35m/setup.py`](training_sd35m/setup.py) for the full install (`cd training_sd35m && pip install -e .`).

### 2. Build training pairs

```bash
cd data_curation
# follow data_curation/README.md: extract → construct_pairs → score → filter
```

The pipeline outputs a CSV that both training stages consume as `csv_file_path` (SD-1.5) / `train_dataset` (SD-3.5-M). See [`data_curation/README.md`](data_curation/README.md).

### 3. Train

| Model | Stage 1 (Diffusion-DRO) | Stage 2 (Diffusion-DPO, LoRA-init) |
|---|---|---|
| **SD-1.5** | `bash training_sd15/stage1_diffusion_dro/train-irl.sh` | `bash training_sd15/stage2_dpo/lora_init.sh` |
| **SD-3.5-M** | `bash training_sd35m/scripts/single_node/inverse_reinforcement_learning.sh` | `bash training_sd35m/scripts/single_node/dpo.sh` |

Both stages read the same CSV. Stage 2 reads Stage 1's LoRA from `pretrained_lora_path` (SD-1.5) / `config.train.lora_path` (SD-3.5-M); edit the shell (SD-1.5) or `config/sd3_5_medium_dpo.py` (SD-3.5-M) to point at Stage 1's checkpoint before launching Stage 2.

Full hyperparameters, launchers, and config schema: [`training_sd15/README.md`](training_sd15/README.md).

### 4. Evaluate

Reward-model evaluation lives in [`training_sd35m/evaluation/`](training_sd35m/evaluation/) for both SD-1.5 and SD-3.5-M.

DPG-Bench generation + evaluation scripts live in [`DPG-Bench/`](DPG-Bench/) (`DPG-Bench-script-sd-v1-5.sh`, `DPG-Bench-script-sd-3-5-medium.sh`).

## 🤗 Acknowledgement

Our codebase references the code from [Diffusion-DRO](https://github.com/basiclab/DiffusionDRO), [Diffusion-DPO](https://github.com/SalesforceAIResearch/DiffusionDPO), and [Flow-GRPO](https://github.com/yifan123/flow_grpo).

We thank the authors for releasing their implementations.

## ⭐ Citation

> TODO: BibTeX will be added once the ICML 2026 proceedings entry is finalized.
