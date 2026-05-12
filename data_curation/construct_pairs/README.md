# `construct_pairs/` — building fake counterparts of real images

Given the real-image CSV produced by `data_curation/extract/hpdv3.py`, this
folder generates a paired *fake* image for each real image. The four methods
differ only in **how the fake is produced**; the output layout is identical.

| Script | Method | Pipeline | Notes |
|---|---|---|---|
| `inpaint_sdv15.py` | U²-Net saliency + SD v1.5 Inpainting | `StableDiffusionInpaintPipeline` | **Main paper method** |
| `inpaint_sd35m.py` | U²-Net saliency + SD-3.5-M Inpainting | `StableDiffusion3InpaintPipeline` | Rebuttal: stronger inpainter (fp16) |
| `inpaint_pixart.py` | U²-Net saliency + PixArt-α Inpainting | `PixArtAlphaInpaintPipeline` | Rebuttal: DiT-based inpainter (fp16, custom dep) |
| `text2image.py` | Plain T2I (no inpainting) | `StableDiffusionPipeline` | Rebuttal: global vs localized contrast |

## Output layout

Every script writes the same directory structure under `--output_dir`:

```
<output_dir>/
├── real/{uid}.png    # 512×512 center-cropped real image
├── fake/{uid}.png    # method-specific fake counterpart
└── mask/{uid}.png    # binarized saliency mask (only for inpaint_*.py; T2I omits)
```

Resume is automatic: each script scans `fake/` at startup and skips already-
generated UIDs. Combined with `--start_index` / `--end_index` this lets you
shard one CSV across multiple GPUs (see the launcher example below).

## Setup — common to all four methods

Apart from `torch` + `diffusers`, the three saliency-guided variants require
the U²-Net checkpoint:

```bash
mkdir -p model_ckpts
wget -O model_ckpts/u2net.pth \
    https://github.com/xuebinqin/U-2-Net/releases/download/.../u2net.pth
# (see U^2-Net upstream for the exact release URL)
```

Pass the path via `--u2net_ckpt`. The U²-Net model definition itself is
vendored at `u2net_arch/` (Apache-2.0, attribution preserved).

## Setup — PixArt-α specific

`PixArtAlphaInpaintPipeline` is **not** in upstream `diffusers`. Before
running `inpaint_pixart.py` you need one of:

- **Clone PixArt-α upstream and add it to `PYTHONPATH`:**
  ```bash
  git clone https://github.com/PixArt-alpha/PixArt-alpha.git
  export PYTHONPATH="$PWD/PixArt-alpha/scripts:$PYTHONPATH"
  ```
  Then change the import in `inpaint_pixart.py` from
  `from diffusers import PixArtAlphaInpaintPipeline` to
  `from pipeline_pixart_inpaint import PixArtAlphaInpaintPipeline`.

- **Or install a `diffusers` fork** that registers the pipeline under its
  package namespace, in which case the existing import works as-is.

The PixArt-α base checkpoint (`PixArt-alpha/PixArt-XL-2-1024-MS`) is fetched
from the Hugging Face Hub on first run; no manual download needed.

## Running

Single-GPU example (main method):
```bash
python inpaint_sdv15.py \
    --input_dir   /path/to/HPDv3/real \
    --output_dir  /path/to/u2net_next_inpainting/HPDv3 \
    --prompt_file /path/to/HPDv3/real_images_uid_prompt.csv \
    --u2net_ckpt  model_ckpts/u2net.pth
```

Multi-GPU sharding (8 GPUs, 1400 uids each):
```bash
CHUNK=1400
for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python inpaint_sdv15.py \
        --input_dir   ... --output_dir ... --prompt_file ... \
        --u2net_ckpt  model_ckpts/u2net.pth \
        --start_index $((i * CHUNK)) --end_index $(((i + 1) * CHUNK)) &
done
wait
```

## Pipeline placement

```
data_curation/
├── extract/hpdv3.py            # writes real_images_uid_prompt.csv
├── construct_pairs/            # ← you are here, writes real/ + fake/ + mask/
├── score/                      # next step: per-pair colorfulness, PickScore,
│                               #            and (HPDv3) anime classification
└── filter/                     # then: anime drop → color filter →
                                #       discard negative → top-512
```
