"""Construct fake counterparts via plain Text-to-Image (no inpainting).

Rebuttal variant: instead of regenerating the salient region of the real
image, generate a fully new image from the same prompt. The real image is
loaded only so that ``real/{uid}.png`` is saved next to ``fake/{uid}.png``
with the same uid index — the pipeline itself never sees it.

This serves as a baseline contrast where positive (real) and negative (T2I)
samples differ globally, not just in a localized salient region.
"""

import argparse
import os
import sys

import torch
from diffusers import StableDiffusionPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner import run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_dir", required=True, help="Directory of real images (kept for output pairing).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--start_index", type=int, default=None)
    p.add_argument("--end_index", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(leave=False)

    def inference(real_image, prompt):
        fake = pipeline(prompt, num_inference_steps=args.num_inference_steps).images[0]
        return {"fake": fake}  # no mask: T2I does not use the real image

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
