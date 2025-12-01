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
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
import pandas as pd


def main(args):
    device = torch.device("cuda")    
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model and Pipeline ---
    print("Loading model and pipeline (runwayml/stable-diffusion-v1-5)...")
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    df = pd.read_csv(args.prompt_file, dtype=str)
    
    # Apply start_index and end_index if provided
    start_idx = args.start_index if args.start_index is not None else 0
    end_idx = args.end_index if args.end_index is not None else len(df)
    df = df.iloc[start_idx:end_idx]
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating images"):
        uid = row["uid"]
        prompt = row["prompt"]
        image = pipeline(prompt).images[0]
        image.save(os.path.join(args.output_dir, f"{uid}.png"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion v1.5.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/text-to-image/HPDv3/no_anime-hpdv3_all/fake",
        help="Directory to save evaluation results and generated images.",
    )   
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/no_anime_all_images.csv",
        help="CSV file stores the uid and corresponding prompt.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index for processing CSV rows (0-based). If not specified, starts from the beginning.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for processing CSV rows (exclusive). If not specified, processes until the end.",
    )

    args = parser.parse_args()
    main(args)

# python text-to-image-sdv1-5.py --output_dir /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/text-to-image/HPDv3/no_anime-hpdv3_all/fake --prompt_file /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/no_anime_all_images.csv