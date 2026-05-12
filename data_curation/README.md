# `data_curation/` — building the (real, fake) preference dataset

RealAlign trains on preference pairs of the form *(reference image,
inpainted fake)*. The reference is the positive anchor — either a real
photo (HPDv3) or a high-quality generated image (Pick-a-Pic v2 /
Civitai-top). This folder builds those pairs from a reference-image source
and curates them into a high-quality training set.

## Reference image sources

The paper draws (positive) reference anchors from three sources. All three
go through the same `construct_pairs/` and `score/` steps; they differ in
the *type* of reference image and in which curation steps they need at the
`filter/` stage:

| Source | Reference image type | Curation steps |
|---|---|---|
| **HPDv3** (`real_images` split) | real (photographic) | `anime drop` → `color filter` → `discard negative` → `top-512` |
| **Pick-a-Pic v2** | generated | `discard negative` → `top-512` |
| **Civitai-top** | generated | `discard negative` → `top-512` |

HPDv3's `real_images` split is noisier than the others — it mixes in
illustrations / digital art and has uneven colorfulness — so it gets two
extra cleanup steps (`anime drop` and `color filter`). Pick-a-Pic v2 and
Civitai-top references are already curated for visual quality and content
upstream (community ratings / preference annotations), so those steps are
skipped.

Even when the reference is a *generated* image (Pick-a-Pic v2 / Civitai-top),
the inpainted counterpart is still constructed by the same `construct_pairs/`
pipeline — saliency mask + re-inpainting — and serves as the negative in the
preference pair.

## Pipeline

```
extract/        →  construct_pairs/   →  score/                    →  filter/
real uid+prompt    real / fake / mask     colorfulness, pickscore     per-source
CSV                triples on disk        CSVs                        curation steps
```

1. **`extract/hpdv3.py`** — parse HPDv3 `all.json` and dump `(uid, prompt)`
   for the `real_images` split into a CSV. (Pick-a-Pic v2 and Civitai-top
   each have their own extraction logic — not yet checked in.)
2. **`construct_pairs/`** — for each reference real image, generate a fake
   counterpart (saliency-guided inpainting; four method variants). See
   [`construct_pairs/README.md`](construct_pairs/README.md).
3. **`score/`** — score every pair (and classify references, for HPDv3):
   - `colorfulness.py` — Hasler & Süsstrunk colorfulness on
     `(reference, fake)` (HPDv3 only).
   - `pickscore.py` — PickScore on `(reference, prompt)` and
     `(fake, prompt)`.
   - `anime.py` — Qwen3-VL artwork/anime classifier on the reference image
     (HPDv3 only).
4. **`filter/`** — apply the per-source curation steps and select the final
   training set. See [`filter/README.md`](filter/README.md).

## The four curation primitives

### `anime drop` (HPDv3 only)

Drop rows where the reference image was classified as artwork or anime
style by `score/anime.py`. HPDv3's `real_images` split mixes in
illustrations and digital art that aren't photographic; these break the
"real anchor" assumption of the paper, so they're removed before any
per-pair filtering. Pick-a-Pic v2 / Civitai-top references are already
known to be photo-style (or stylistically consistent) and don't need this.

### `color filter` (HPDv3 only)

Keep pairs where `colorfulness(reference) > colorfulness(fake)`. HPDv3
`real_images` contains many low-colorfulness photos; this drops cases where
the inpainter actually produced something *more* colorful than the
reference (which would invert the preference signal in colorfulness-
sensitive metrics).

### `discard negative`

Keep pairs where `PickScore(reference, prompt) - PickScore(fake, prompt) >
0.02`. A non-positive or small gap means the prompt aligns with the fake
better than with the reference — usually because the source prompt drifted
from the photo content. Such pairs would teach the model the wrong
direction.

### `top-512` selection

After the previous filter(s), sort the surviving pairs by
`PickScore(reference)` (the reference image's PickScore alone, not the
gap) descending, and take the top 512 as the training set. Sorting by the
reference score keeps the highest-quality anchors. Paper Figure 7 ablates
256 / 512 / 768 / 1024; 512 is the default.

## Layout

```
data_curation/
├── README.md                      # this file
├── extract/
│   └── hpdv3.py
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
