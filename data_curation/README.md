# `data_curation/` — building the (real, fake) preference dataset

RealAlign is trained on **(reference, fake)** preference pairs, where the reference is the positive anchor — either a real photograph (HPDv3) or a high-quality generated image (Civitai-top, or top-PickScore from Pick-a-Pic v2). This folder produces such pairs from a reference-image source and curates them into a high-quality training set.

## 🖼️ Reference sources

| Source | Reference type | HuggingFace repo |
|---|---|---|
| **HPDv3** (`real_images` split) | real photograph | 🤗 [`MizzenAI/HPDv3`](https://huggingface.co/datasets/MizzenAI/HPDv3) |
| **Civitai-top** | generated | 🤗 [`wallstoneai/civitai-top-sfw-images-with-metadata`](https://huggingface.co/datasets/wallstoneai/civitai-top-sfw-images-with-metadata) |
| **Pick-a-Pic v2** | generated | 🤗 [`liuhuohuo2/pick-a-pic-v2`](https://huggingface.co/datasets/liuhuohuo2/pick-a-pic-v2) |

All three sources flow through the same `extract → construct_pairs → score → filter` pipeline; they differ only in which curation steps apply at the final `filter/` stage.

## 🔄 Pipeline

Four stages, run in order. Each stage's output is the next stage's input.

### 1. Extract — `extract/`

Parse the source dataset and emit a `(uid, prompt)` CSV.

| Source | Script | Notes |
|---|---|---|
| HPDv3 | `extract/hpdv3.py` | filters `all.json` to entries whose source is `real_images`; writes `real_images_uid_prompt.csv`. |
| Civitai-top | `extract/civitai_top.py` | parses `prompts.json`, uses each image's filename stem as `uid`; writes `uid_prompt.csv`. |
| Pick-a-Pic v2 | — | skipped here; the dataset is too large for this lightweight extraction flow. |

### 2. Construct pairs — `construct_pairs/`

For each `(uid, prompt)` row in the CSV emitted by Stage 1, generate the **fake** counterpart (the reference image is loaded from the source dataset using `uid`). Two modes:

- **Inpainting** — U²-Net localizes the salient region of the reference; a generator fills it back in.
  - `inpaint_sdv15.py` — `StableDiffusionInpaintPipeline`. **Main paper method.**
  - `inpaint_pixart.py` — `PixArtAlphaInpaintPipeline`. DiT-based variant.
  - `inpaint_sd35m.py` — `StableDiffusion3InpaintPipeline`. Stronger SD-3.5-M variant.
- **Text-to-image** — `text2image.py` generates from the prompt directly, with no saliency mask.

### 3. Score — `score/`

Compute the per-pair signals that the `filter/` stage will threshold on.

| Script | Signal |
|---|---|
| `colorfulness.py` | Hasler & Süsstrunk colorfulness of the reference image. |
| `pickscore.py` | `yuvalkirstain/PickScore_v1` alignment for `(image, prompt)`. |
| `anime.py` | Qwen3-VL-8B-Instruct flag for non-photographic references (HPDv3 only). |

### 4. Filter — `filter/`

Apply source-specific curation to produce the final training CSV.

| Source | Script | Steps (in order) |
|---|---|---|
| HPDv3 | `filter/hpdv3.py` | drop non-photographic → 🎨 colorfulness → 🔎 negligible-degradation → 🏆 top-512 |
| Pick-a-Pic v2 / Civitai-top | `filter/external.py` | 🔎 negligible-degradation → 🏆 top-512 |

The four curation steps:

- **Drop non-photographic** *(HPDv3 only)* — discard rows that `score/anime.py` flags as artwork / anime. HPDv3's `real_images` split mixes in illustrations and digital art; this strips them so only real photographs survive.
- **🎨 Colorfulness filter** *(HPDv3 only)* — keep pairs where `colorfulness(reference) > mean(colorfulness over the reference set)`. A lightweight heuristic against visually flat or low-contrast references.
- **🔎 Negligible-degradation filter** — keep pairs where `PickScore(reference, prompt) − PickScore(fake, prompt) > 0.02`. Ensures every pair carries a clear and consistent preference signal.
- **🏆 Top-512 by PickScore** — sort surviving pairs by `PickScore(reference)` descending and take the top 512. Paper Figure 7 ablates 256 / 512 / 768 / 1024; **512 is the default**.

## 📁 Layout

```
data_curation/
├── README.md                      # this file
├── extract/
│   ├── hpdv3.py
│   └── civitai_top.py
├── construct_pairs/
│   ├── README.md
│   ├── runner.py / saliency.py
│   ├── inpaint_{sdv15, sd35m, pixart}.py
│   ├── text2image.py
│   └── u2net_arch/                # vendored U^2-Net (Apache-2.0)
├── score/
│   ├── colorfulness.py
│   ├── pickscore.py
│   └── anime.py
└── filter/
    ├── README.md
    ├── hpdv3.py                   # 4 steps
    └── external.py                # 2 steps (Pick-a-Pic v2 / Civitai-top)
```
