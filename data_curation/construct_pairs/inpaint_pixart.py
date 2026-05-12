"""Construct fake counterparts via U^2-Net saliency + PixArt-alpha Inpainting.

Rebuttal variant: replaces the SD v1.5 inpainter with PixArt-alpha (DiT-based).
The base model is the 1024-MS checkpoint; we still feed 512-resolution real
images and pass explicit ``height/width`` to the pipeline so the inpainter
operates at the input resolution.

Dependency note
---------------
``PixArtAlphaInpaintPipeline`` is **not** part of upstream ``diffusers``.
Before running this script you need to either:

  (a) clone https://github.com/PixArt-alpha/PixArt-alpha and add
      ``PixArt-alpha/scripts`` to your ``PYTHONPATH`` so that
      ``pipeline_pixart_inpaint.py`` is importable, then change the import
      below to ``from pipeline_pixart_inpaint import PixArtAlphaInpaintPipeline``;

  (b) install a diffusers fork that registers ``PixArtAlphaInpaintPipeline``
      under ``diffusers``.

See ``data_curation/construct_pairs/README.md`` for setup details.
"""

import argparse
import os
import sys

import torch
from diffusers import PixArtAlphaInpaintPipeline  # see dependency note

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner import run
from saliency import U2NetSaliency


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--u2net_ckpt", required=True)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--start_index", type=int, default=None)
    p.add_argument("--end_index", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    saliency = U2NetSaliency(args.u2net_ckpt, device=device)

    pipeline = PixArtAlphaInpaintPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    ).to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(leave=False)

    def inference(real_image, prompt):
        mask = saliency(real_image)
        fake = pipeline(
            prompt,
            image=real_image,
            mask_image=mask,
            strength=1.0,
            height=real_image.height,
            width=real_image.width,
            num_inference_steps=args.num_inference_steps,
        ).images[0]
        return {"fake": fake, "mask": mask}

    run(
        inference_fn=inference,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        start_index=args.start_index,
        end_index=args.end_index,
    )


if __name__ == "__main__":
    main()
