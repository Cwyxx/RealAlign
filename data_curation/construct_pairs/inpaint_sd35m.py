"""Construct fake counterparts via U^2-Net saliency + SD-3.5-Medium Inpainting.

Rebuttal variant: same saliency-guided inpainting setup as the SD v1.5 main
method, but with ``StableDiffusion3InpaintPipeline`` swapped in to test
whether a stronger inpainter changes the trend. fp16 + ``strength=1.0`` to
match the upstream defaults.
"""

import argparse
import os
import sys

import torch
from diffusers import StableDiffusion3InpaintPipeline

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

    pipeline = StableDiffusion3InpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.float16,
    ).to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(leave=False)

    def inference(real_image, prompt):
        mask = saliency(real_image)
        fake = pipeline(
            prompt=prompt,
            image=real_image,
            mask_image=mask,
            strength=1.0,
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
