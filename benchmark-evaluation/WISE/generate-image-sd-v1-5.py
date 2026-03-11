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


class WISEPromptDataset(Dataset):
    """WISE benchmark: load JSON from data_dir by dataset type (wise=.json, wise-rewrite=_rewrite.json), merge and sort by prompt_id (1-1000)."""
    def __init__(self, data_dir, dataset="wise"):
        self.data_dir = os.path.abspath(data_dir)
        self.dataset = dataset  # "wise" -> .json (exclude _rewrite); "wise-rewrite" -> _rewrite.json only
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"WISE data directory not found: {self.data_dir}")
        if dataset not in ("wise", "wise-rewrite"):
            raise ValueError(f"dataset must be 'wise' or 'wise-rewrite', got: {dataset}")

        use_rewrite = dataset == "wise-rewrite"
        items = []
        for name in sorted(os.listdir(self.data_dir)):
            if use_rewrite:
                if not name.endswith("_rewrite.json"):
                    continue
            else:
                if not name.endswith(".json") or name.endswith("_rewrite.json"):
                    continue
            path = os.path.join(self.data_dir, name)
            if not os.path.isfile(path):
                continue
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded {len(data)} items from {path}")
            if isinstance(data, list):
                items.extend(data)
            else:
                items.append(data)
        items.sort(key=lambda x: x.get("prompt_id", 0))
        self.prompts = [x["Prompt"] for x in items]
        self.prompt_ids = [x["prompt_id"] for x in items]
        self.metadatas = items
        assert len(self.prompts) == 1000, f"WISE expects 1000 prompts, got {len(self.prompts)}"

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "metadata": self.metadatas[idx],
            "original_index": self.prompt_ids[idx],
        }


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    indices = [example["original_index"] for example in examples]
    return prompts, metadatas, indices


def main(args):
    device = torch.device("cuda")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")

    # --- Load Model and Pipeline ---
    unet_init = args.unet_init
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
    json_data_dir = args.json_data_dir
    if not json_data_dir:
        json_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    suffix_hint = "_rewrite.json" if args.dataset == "wise-rewrite" else ".json (excluding *_rewrite.json)"
    print(f"Loading WISE dataset from: {json_data_dir} (suffix: {suffix_hint})")
    dataset = WISEPromptDataset(json_data_dir, dataset=args.dataset)
            
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
        "--json_data_dir",
        type=str,
        default=None,
        help="[WISE only] Directory containing WISE JSON files. Default: <script_dir>/data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wise",
        choices=["wise", "wise-rewrite"],
        help="Which JSON to load: 'wise' = .json (exclude *_rewrite.json); 'wise-rewrite' = *_rewrite.json only.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3",
        help="Directory to save evaluation results and generated images.",
    )   
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the generated images.")
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
        default="runwayml/stable-diffusion-v1-5",
        choices=["mhdang/dpo-sd1.5-text2image-v1", "jacklishufan/diffusion-kto", "ylwu/diffusion-dro-sd1.5", "runwayml/stable-diffusion-v1-5", "JaydenLu666/InPO-SD1.5"],
        help="Unet initialization model.",
    )
    args = parser.parse_args()
    main(args)
