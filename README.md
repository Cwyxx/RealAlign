<h1 align="center">When Preference Labels Fall Short:<br>Aligning Diffusion Models from Real Data</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.19839"><img src="https://img.shields.io/badge/%F0%9F%93%84_Paper-arXiv:2605.19839-red" alt="Paper"></a>
  <a href="TODO"><img src="https://img.shields.io/badge/%F0%9F%93%96_OpenReview-Forum-red" alt="OpenReview"></a>
  <a href="https://cwyxx.github.io/RealAlign/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Project_WebPage-green" alt="Project Page"></a>
  <a href="https://github.com/Cwyxx/RealAlign"><img src="https://img.shields.io/badge/%F0%9F%9A%80_Code-GitHub-blue" alt="Code"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/Xixixixihahahaha/RealAlign-SD-1.5"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model_%28SD1.5%29-HuggingFace-yellow" alt="Model SD-1.5"></a>
  <a href="https://huggingface.co/Xixixixihahahaha/RealAlign-SD-3.5-M"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model_%28SD3.5M%29-HuggingFace-yellow" alt="Model SD-3.5-M"></a>
  <a href="https://huggingface.co/datasets/Xixixixihahahaha/RealAlign-Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Dataset-HuggingFace-yellow" alt="Dataset"></a>
</p>


## 📝 Abstract

Preference alignment aims to guide generative models by learning from comparisons between preferred and non-preferred samples. In practice, most existing approaches rely on preference pairs constructed from model-generated images. Such supervision is inherently relative and can be ambiguous when both samples exhibit artifacts or limited visual quality, making it difficult to infer what constitutes a truly desirable output. In this work, we investigate whether real data can serve as an alternative source of supervision for preference alignment. We adopt a data-centric perspective and study a curation strategy that treats real images as reference points and constructs preference signals by contrasting them with generated or perturbed samples, without requiring manually annotated preference pairs. Through empirical analysis, we show that real-data-based supervision provides effective guidance for aligning diffusion models and achieves performance comparable to existing preference-based methods. Our results suggest that real data offers a practical and complementary source of supervision for preference alignment and highlight directions of label-efficient alignment strategies.

## 🖼️ Gallery

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="assets/gallery/00009.png" alt="gallery 00009" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00149.png" alt="gallery 00149" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00269.png" alt="gallery 00269" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00451.png" alt="gallery 00451" width="100%"/></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="assets/gallery/00014.png" alt="gallery 00014" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00186.png" alt="gallery 00186" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00150.png" alt="gallery 00150" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00243.png" alt="gallery 00243" width="100%"/></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="assets/gallery/00043.png" alt="gallery 00043" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00198.png" alt="gallery 00198" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00264.png" alt="gallery 00264" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/01175.png" alt="gallery 01175" width="100%"/></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="assets/gallery/00060.png" alt="gallery 00060" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00654.png" alt="gallery 00654" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00368.png" alt="gallery 00368" width="100%"/></td>
    <td align="center" width="25%"><img src="assets/gallery/00078.png" alt="gallery 00078" width="100%"/></td>
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

**Evaluation**

- [`training_sd35m/evaluation/`](training_sd35m/evaluation/) contains the reward-model evaluation harness for both SD-1.5 and SD-3.5-M.
  - [`training_sd35m/evaluation/sd-v1-5/`](training_sd35m/evaluation/sd-v1-5/) provides SD-1.5 `generate_image.py`, `calculate_score.py`, and `run_multi_seed_eval.sh`.
  - [`training_sd35m/evaluation/sd-3-5-medium/`](training_sd35m/evaluation/sd-3-5-medium/) provides SD-3.5-M `generate_image.py`, `calculate_score.py`, and `run_multi_seed_eval.sh`.
  - Prompt lists live under `training_sd35m/dataset/`, including `pick_a_pic_v2/`, `partiprompts/`, and `drawbench-unique/`.
  - All six reward metrics, PickScore, ImageReward, Aesthetic, HPSv3, DeQA, and UnifiedReward, are routed through `flow_grpo.rewards.multi_score`.
- [`DPG-Bench/`](DPG-Bench/) contains DPG-Bench evaluation scripts for SD-1.5 and SD-3.5-M.

**Paper artifacts**

- [`notebook/`](notebook/) contains curated ICML 2026 analysis notebooks and figure artifacts used in the paper.

## 🚀 Quick start

### 1. Environment Set Up

Create and activate the shared `alignprop` environment:

```bash
conda create -n alignprop python=3.10
conda activate alignprop
pip install -r requirements.txt
```

### 2. Build training pairs

Run the four-stage pipeline in [`data_curation/`](data_curation/) (`extract → construct_pairs → score → filter`) to produce the paired `(real, fake)` training CSV. Full details: [`data_curation/README.md`](data_curation/README.md).

### 3. Train

**SD-1.5**

```bash
# Stage 1: Diffusion-DRO
bash training_sd15/stage1_diffusion_dro/train-irl.sh
# Stage 2: Diffusion-DPO (LoRA-init)
bash training_sd15/stage2_dpo/lora_init.sh
```

**SD-3.5-M**

```bash
# Stage 1: Diffusion-DRO
bash training_sd35m/scripts/single_node/inverse_reinforcement_learning.sh
# Stage 2: Diffusion-DPO (LoRA-init)
bash training_sd35m/scripts/single_node/dpo.sh
```

- **Shared input.** Both stages read the same CSV — `csv_file_path_train` for SD-1.5, `config.{irl,dpo}.csv_file_path` for SD-3.5-M. SD-3.5-M also expects precomputed prompt embeddings (see [`training_sd35m/README.md`](training_sd35m/README.md)).
- **Stage 2 warm-starts from Stage 1.** Before launching Stage 2, point it at the Stage 1 LoRA checkpoint: set `pretrained_lora_path` in [`training_sd15/stage2_dpo/lora_init.sh`](training_sd15/stage2_dpo/lora_init.sh) (SD-1.5) or `config.train.lora_path` in [`training_sd35m/config/sd3_5_medium_dpo.py`](training_sd35m/config/sd3_5_medium_dpo.py) (SD-3.5-M).

Full hyperparameters, launchers, and config schema: [`training_sd15/README.md`](training_sd15/README.md), [`training_sd35m/README.md`](training_sd35m/README.md).

### 4. Evaluate

Reward-model evaluation lives in [`training_sd35m/evaluation/`](training_sd35m/evaluation/) for both SD-1.5 and SD-3.5-M.

DPG-Bench generation + evaluation scripts live in [`DPG-Bench/`](DPG-Bench/) (`DPG-Bench-script-sd-v1-5.sh`, `DPG-Bench-script-sd-3-5-medium.sh`).

## 🙏 Acknowledgement

Our codebase references the code from [Diffusion-DRO](https://github.com/basiclab/DiffusionDRO), [Diffusion-DPO](https://github.com/SalesforceAIResearch/DiffusionDPO), and [Flow-GRPO](https://github.com/yifan123/flow_grpo).

We thank the authors for releasing their implementations.

## 📚 Citation

> TODO: BibTeX will be added once the ICML 2026 proceedings entry is finalized.
