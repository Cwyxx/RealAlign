"""Shared loop for fake-counterpart construction.

All four ``construct_pairs`` methods (SD v1.5 / SD-3.5-M / PixArt-alpha
inpainting + plain text-to-image) share the same outer loop:

    for each (uid, prompt) row in the input CSV:
        load <input_dir>/{uid}.{ext}
        center-crop to a square at the target resolution
        call inference_fn(real_image, prompt) -> dict with at least "fake"
        save real/{uid}.png, fake/{uid}.png, optionally mask/{uid}.png

Method-specific differences (which pipeline to load, whether saliency is used,
extra pipeline kwargs like ``strength``) live entirely inside ``inference_fn``.

Resume support: the loop scans ``output_dir/fake/`` at startup and skips
already-generated UIDs. Combined with ``start_index`` / ``end_index``, the same
script can be sharded across multiple GPUs (see the launcher .sh examples).
"""

import os
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from saliency import binarize_mask


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".PNG", ".JPEG", ".JPG")


def _find_image(input_dir: str, uid: str) -> Optional[str]:
    for ext in _IMG_EXTS:
        path = os.path.join(input_dir, f"{uid}{ext}")
        if os.path.exists(path):
            return path
    return None


def run(
    *,
    inference_fn: Callable[[Image.Image, str], dict],
    input_dir: str,
    output_dir: str,
    prompt_file: str,
    resolution: int = 512,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> None:
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Input images:  {input_dir}")
    print(f"Prompt file:   {prompt_file}")
    print(f"Output:        {output_dir}")

    processed_uids = {
        os.path.splitext(f)[0]
        for f in os.listdir(fake_dir) if f.endswith(".png")
    }
    print(f"Already processed: {len(processed_uids)} uids")

    df = pd.read_csv(prompt_file, dtype={"uid": str})
    total = len(df)
    if start_index is not None or end_index is not None:
        s = start_index or 0
        e = end_index if end_index is not None else total
        df = df.iloc[s:e]
        print(f"Shard range:   [{s}:{e}] of {total} -> {len(df)} items")

    crop = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
    ])

    n_done, n_skip = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="construct_pairs"):
        uid, prompt = row["uid"], row["prompt"]
        if uid in processed_uids:
            n_skip += 1
            continue

        img_path = _find_image(input_dir, uid)
        if img_path is None:
            print(f"\nWarning: no image for uid {uid}, skipping")
            continue

        try:
            real_image = crop(Image.open(img_path).convert("RGB"))
            outputs = inference_fn(real_image, prompt)
        except Exception as e:
            print(f"\nError on uid {uid}: {e}")
            continue

        real_image.save(os.path.join(real_dir, f"{uid}.png"))
        outputs["fake"].save(os.path.join(fake_dir, f"{uid}.png"))
        if "mask" in outputs and outputs["mask"] is not None:
            binarize_mask(outputs["mask"]).save(os.path.join(mask_dir, f"{uid}.png"))
        n_done += 1

    print(f"\nDone. Processed: {n_done}, skipped (already done): {n_skip}, range total: {len(df)}")
