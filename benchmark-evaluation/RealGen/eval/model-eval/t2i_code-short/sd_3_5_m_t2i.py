#!/usr/bin/env python

import argparse
import os
import json
import logging

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class PromptDataset(Dataset):
    """Load prompts from a text file (one prompt per line)."""
    def __init__(self, file_path):
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            self.prompts = [
                'portrait of a woman',
                'A realistic portrait of a young woman, clear facial features, smooth black hair, natural makeup, wearing casual clothes, photographed in soft natural light.',
            ]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def main(args):
    device = torch.device("cuda")

    # Mixed Precision Setup
    mixed_precision_dtype = None
    if args.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None

    print("Running RealGen evaluation with SD-3.5-M on 1 GPU.")
    if enable_amp:
        print(f"Using mixed precision: {args.mixed_precision}")

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Load Model and Pipeline
    print("Loading model and pipeline (stabilityai/stable-diffusion-3.5-medium)...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium"
    )

    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(
        r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"args.checkpoint_path: {args.checkpoint_path}")
    if (
        args.checkpoint_path is not None
        and args.checkpoint_path
        and os.path.exists(os.path.join(args.checkpoint_path, "lora", "learner"))
    ):
        lora_path = os.path.join(args.checkpoint_path, "lora", "learner")
        print(f"Loading LoRA weights from: {lora_path}")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"LoRA directory not found at {lora_path}. Ensure your checkpoint has a 'lora' subdirectory."
            )

        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.load_adapter(
            lora_path, adapter_name="learner", is_trainable=False
        )
        pipeline.transformer.set_adapter("learner")
        print(f"pipeline.transformer.active_adapter: {pipeline.transformer.active_adapter}")

    pipeline.transformer.eval()
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.transformer.to(device, dtype=text_encoder_dtype)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Load Dataset
    print(f"Loading prompts from: {args.prompt_file}")
    dataset = PromptDataset(args.prompt_file)
    print(f"Total prompts: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # Generate images
    for idx, prompt in enumerate(tqdm(dataloader, desc="Generating images")):
        prompt = prompt[0]
        generator = torch.Generator(device).manual_seed(args.seed + idx * 1000)

        with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
            with torch.no_grad():
                image = pipeline(
                    prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pt",
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator,
                )[0][0]

        image_path = os.path.join(images_dir, f"{idx:05d}.png")
        pil_image = Image.fromarray(
            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        pil_image.save(image_path)

    print(f"Generation completed! Images saved to: {images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SD-3.5-M images for RealGen benchmark"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to SD3 LoRA checkpoint directory (expects '<checkpoint_path>/lora/learner')"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to text file containing prompts (one per line)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of diffusion inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of generated images"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision"
    )
    args = parser.parse_args()
    main(args)
