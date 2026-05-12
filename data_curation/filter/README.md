# `filter/` — selecting the final 512 training pairs

After `score/` has produced per-pair score CSVs, this folder applies the
curation steps and selects the final training set.

## Per-source pipelines

| Source | Steps | Script |
|---|---|---|
| **HPDv3** | anime drop → color filter → discard negative → top-512 | `hpdv3.py` |
| **Pick-a-Pic v2** | discard negative → top-512 | `external.py` (set paths to Pick-a-Pic v2) |
| **Civitai-top** | discard negative → top-512 | `external.py` (set paths to Civitai-top) |

The four curation primitives:

- **anime drop** (HPDv3 only) — drop rows where the reference image was
  classified as artwork / anime by `score/anime.py`. HPDv3's `real_images`
  split mixes in illustrations and digital art that aren't photographic;
  these break the "real anchor" assumption of the paper, so they're
  removed before any per-pair filtering. Pick-a-Pic v2 / Civitai-top
  references are already known to be photo-style or stylistically
  consistent and don't need this.
- **color filter** (HPDv3 only) — keep pairs where
  `colorfulness(reference) > colorfulness(fake)`. HPDv3's `real_images`
  split has many low-colorfulness photos; this drops cases where the
  reference is visually flatter than its inpainted counterpart.
- **discard negative** — keep pairs where
  `PickScore(reference) - PickScore(fake) > 0.02`. Pairs with a non-positive
  or small gap mean the prompt actually describes the *fake* better than the
  reference; they would teach the wrong preference direction.
- **top-K selection** — sort the surviving pairs by `PickScore(reference)`
  descending and take the top `K` (default 512). The reference image
  is the *positive* anchor, so we keep the highest-scoring anchors.

Pick-a-Pic v2 and Civitai-top skip the anime drop and color filter because
their reference images are already pre-curated for visual quality upstream
(community ratings / preference annotations).

## Usage

Each script is a flat top-level-constants module — edit the paths at the
top of the file, then run:

```bash
python -m data_curation.filter.hpdv3
python -m data_curation.filter.external   # Pick-a-Pic v2 (default paths)
# then edit the constants for Civitai-top and run again
```

## Output schema

Both scripts write a CSV with one row per selected pair:

```
uid, prompt, real_image_path, fake_image_path,
pickscore_real, pickscore_fake[, color_real, color_fake]
```

The `color_*` columns are present only in `hpdv3.py`'s output. The Stage 1 /
Stage 2 dataloaders read this CSV directly — no further preprocessing.
