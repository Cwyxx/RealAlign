"""Construct fake counterparts via U^2-Net saliency + SD v1.5 Inpainting.

This is the **main paper** method: detect the salient region of a real image
with U^2-Net, then regenerate that region under the original prompt using
``StableDiffusionInpaintPipeline``. The unmasked area is preserved, so the
fake differs from the real primarily inside the salient region — exposing
preference-relevant deviations in texture, structure, or semantics.

Usage::

    python inpaint_sdv15.py \\
        --input_dir   /path/to/HPDv3/real \\
        --output_dir  /path/to/u2net_next_inpainting/HPDv3 \\
        --prompt_file /path/to/HPDv3/real_images_uid_prompt.csv \\
        --u2net_ckpt  /path/to/u2net.pth
"""

import argparse
import os
import sys

import torch
from diffusers import StableDiffusionInpaintPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner import run
from saliency import U2NetSaliency


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_dir", required=True, help="Directory of real images.")
    p.add_argument("--output_dir", required=True, help="Where real/fake/mask subdirs are written.")
    p.add_argument("--prompt_file", required=True, help="CSV with columns (uid, prompt).")
    p.add_argument("--u2net_ckpt", required=True, help="Path to U^2-Net weights (u2net.pth).")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--start_index", type=int, default=None, help="Shard start (for parallel runs).")
    p.add_argument("--end_index", type=int, default=None, help="Shard end (exclusive).")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    saliency = U2NetSaliency(args.u2net_ckpt, device=device)

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting"
    ).to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(leave=False)

    def inference(real_image, prompt):
        mask = saliency(real_image)
        fake = pipeline(
            prompt=prompt,
            image=real_image,
            mask_image=mask,
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
