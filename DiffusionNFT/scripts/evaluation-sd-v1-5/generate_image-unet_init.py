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

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torch.utils.data import DataLoader, Dataset


import torch.distributed as dist
from peft import PeftModel

import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}, "original_index": idx}


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}_metadata.jsonl")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx], "original_index": idx}


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    indices = [example["original_index"] for example in examples]
    return prompts, metadatas, indices


def main(args):
    device = torch.device("cuda")
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_images: os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")

    # --- Load Model and Pipeline ---
    # unet_init = "mhdang/dpo-sd1.5-text2image-v1"
    unet_init = args.unet_init
    # unet_init = "ylwu/diffusion-dro-sd1.5"
    print(f"Loading model and pipeline ({unet_init})...")
    unet = UNet2DConditionModel.from_pretrained(unet_init, subfolder="unet")
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.unet = unet
    
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        print(f"Loading LoRA weights from: {args.checkpoint_path}")
        pipeline.load_lora_weights(args.checkpoint_path, weight_name="pytorch_lora_weights.safetensors")
        print(f"Activate LoRA: {pipeline.get_active_adapters()}")
        
    pipeline.to(device)
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

    if args.dataset == "geneval":
        dataset = GenevalPromptDataset(dataset_path, split="test")

    elif args.dataset == "ocr":
        dataset = TextPromptDataset(dataset_path, split="test")

    elif args.dataset == "pickscore":
        dataset = TextPromptDataset(dataset_path, split="test")

    elif args.dataset == "drawbench":
        dataset = TextPromptDataset(dataset_path, split="test")
        
    elif args.dataset == "pick_a_pic_spo":
        dataset = TextPromptDataset(dataset_path, split="test")
        
    elif args.dataset == "pickscore_train":
        dataset_path = f"../dataset/pickscore"
        dataset = TextPromptDataset(dataset_path, split="train")
        
    elif args.dataset == "x_aigd":
        dataset = TextPromptDataset(dataset_path, split="test")
        
    elif args.dataset == "pick_a_pic_v2":
        dataset = TextPromptDataset(dataset_path, split="test")
        
    elif args.dataset == "drawbench-unique":
        dataset = TextPromptDataset(dataset_path, split="test")
    eval_batch_size = 1

    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )
    
    result_this_rank = []
    for batch in tqdm(dataloader, desc=f"Evaluating"):
        prompts, metadata, indices = batch
        current_batch_size = len(prompts)
        generator = torch.Generator(device).manual_seed(args.seed)
        
        with torch.no_grad():
            images = pipeline(
                prompts,
                output_type="pt",
                height=args.resolution,
                width=args.resolution,
                generator=generator
            )[0]

        for i in range(current_batch_size):
            sample_idx = indices[i]
            result_item = {
                "sample_id": sample_idx,
                "prompt": prompts[i],
                "metadata": metadata[i] if metadata else {},
                "scores": {},
            }

            if args.save_images:
                image_path = os.path.join(args.output_dir, "images", f"{sample_idx:05d}.png")
                pil_image = Image.fromarray((images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil_image.save(image_path)
                result_item["image_path"] = image_path

            result_this_rank.append(result_item)
        
    result_this_rank.sort(key=lambda x: x["sample_id"])

    with open(results_filepath, "w") as f_out:
        for result_item in result_this_rank:
            f_out.write(json.dumps(result_item) + "\n")
    
    with open(results_filepath.replace(".jsonl", "_backup.jsonl"), "w") as f_out:
        for result_item in result_this_rank:
            f_out.write(json.dumps(result_item) + "\n")

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
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench", "pick_a_pic_spo", "pickscore_train", "x_aigd", "pick_a_pic_v2", "drawbench-unique"], help="Dataset type."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3",
        help="Directory to save evaluation results and generated images.",
    )   
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
        "--unet_init",
        type=str,
        default="jacklishufan/diffusion-kto",
        choices=["mhdang/dpo-sd1.5-text2image-v1", "jacklishufan/diffusion-kto", "ylwu/diffusion-dro-sd1.5", "runwayml/stable-diffusion-v1-5", ],
        help="Unet initialization model.",
    )
    args = parser.parse_args()
    main(args)
