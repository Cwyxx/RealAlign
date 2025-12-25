# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    device = torch.device("cuda")

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if args.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None

    print(f"Running evaluation with 1 GPUs.")
    if enable_amp: print(f"Using mixed precision: {args.mixed_precision}")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    # --- Load Model and Pipeline ---
    print("Loading model and pipeline (stabilityai/stable-diffusion-3.5-medium)...")

    if args.model_type == "sd3":
        # pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", text_encoder_3=None, tokenizer_3=None)
        pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
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
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
   
    print(f"args.checkpoint_path: {args.checkpoint_path}")
    if args.checkpoint_path is not None and args.checkpoint_path:
        lora_path = os.path.join(args.checkpoint_path, "lora", "learner")
        if os.path.exists(lora_path):
            print(f"Loading LoRA weights from: {lora_path}")
        else:
            raise FileNotFoundError(
                f"LoRA directory not found at {lora_path}. Ensure your checkpoint has a 'lora' subdirectory."
            )

        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.load_adapter(lora_path, adapter_name="learner", is_trainable=False)
        pipeline.transformer.set_adapter("learner")
        print(f"pipeline.transformer.active_adapter: {pipeline.transformer.active_adapter}")

    pipeline.transformer.eval()
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.transformer.to(device, dtype=text_encoder_dtype)
    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
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

    # --- Load Dataset with Distributed Sampler ---
    dataset_path = f"../../dataset/{args.dataset}"
    print(f"Loading dataset from: {dataset_path}")

    with open(os.path.join(dataset_path, "mscoco_val_2014_10k.json"), "r") as f:
        full_data = json.load(f)
        if args.end_index is None:
            data = full_data[args.start_index:]
            end_idx = len(full_data)
        else:
            data = full_data[args.start_index:args.end_index]
            end_idx = args.end_index
    for item in tqdm(data, desc=f"Evaluating {args.start_index} to {end_idx}"):
        uid = item["image"]
        prompt = item["text"]
        generator = torch.Generator(device).manual_seed(args.seed)
        
        with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
            with torch.no_grad():
                image = pipeline(
                    [prompt],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pt",
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator
                )[0][0]

                if args.save_images:
                    image_path = os.path.join(args.output_dir, "images", f"{uid}.png")
                    pil_image = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil_image.save(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion model in a distributed manner.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Setting this ensures consistent results across runs."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Local path to the LoRA checkpoint directory (e.g., './save/run_name/checkpoints/checkpoint-5000').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["sd3"],
        help="Type of the base model ('sd3').",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench", "pick_a_pic_spo", "pickscore_train", "pick_a_pic_v2", "drawbench_realistic_style", "OneIG-Bench-Anime", "OneIG-Bench-Portrait", "coco"], help="Dataset type."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3",
        help="Directory to save evaluation results and generated images.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=40, help="Number of inference steps for the diffusion pipeline."
    )
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Classifier-free guidance scale.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the generated images.")
    parser.add_argument(
        "--save_images", action="store_true", help="Include this flag to save generated images to the output directory."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between 'no', 'fp16', or 'bf16'.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for dataset slicing.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for dataset slicing. If None, processes all remaining items.",
    )

    args = parser.parse_args()
    main(args)
