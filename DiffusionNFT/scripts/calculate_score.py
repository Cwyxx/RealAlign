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
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from flow_grpo.rewards import multi_score

from collections import defaultdict
from peft import PeftModel

import logging


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
    base_image_dir = os.path.join(args.output_dir, "images")
    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")

    # --- Load Dataset with Distributed Sampler ---
    dataset_path = f"dataset/{args.dataset}"

    if args.dataset == "geneval":
        dataset = GenevalPromptDataset(dataset_path, split="test")
    elif args.dataset == "ocr":
        dataset = TextPromptDataset(dataset_path, split="test")
    elif args.dataset == "pickscore":
        dataset = TextPromptDataset(dataset_path, split="test")
    elif args.dataset == "drawbench":
        dataset = TextPromptDataset(dataset_path, split="test")
    eval_batch_size = 1
    
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # --- Instantiate Reward Models ---
    print("Initializing reward models...")
    if args.reward_model in ["imagereward", "pickscore", "aesthetic", "clipscore", "hpsv2"]:
        all_reward_scorers = {
            args.reward_model: 1.0
        }
        scoring_fn, _ = multi_score(device, all_reward_scorers)
        print(f"Initializing reward models {args.reward_model} from DiffusionNFT...")
    elif args.reward_model == "vqascore":
        import t2v_metrics
        clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl')
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            scores = clip_flant5_score(images=images, texts=prompts)
            diagonal_scores = [scores[i][i] for i in range(len(scores))]
            score_details = { args.reward_model: diagonal_scores}
            return score_details, {}

    # --- Evaluation Loop ---
    results_this_rank = []

    for batch in tqdm(dataloader, desc=f"Evaluating"):
        prompts, metadata, indices = batch
        current_batch_size = len(prompts)

        all_scores, _ = scoring_fn(images, prompts, metadata, only_strict=False)

        for i in range(current_batch_size):
            sample_idx = indices[i]
            result_item = {
                "sample_id": sample_idx,
                "prompt": prompts[i],
                "metadata": metadata[i] if metadata else {},
                "scores": {},
            }

            if args.save_images:
                image_path = os.path.join(args.output_dir, "images", f"{sample_idx:05d}.jpg")
                pil_image = Image.fromarray((images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil_image.save(image_path)
                result_item["image_path"] = image_path

            for score_name, score_values in all_scores.items():
                if isinstance(score_values, torch.Tensor):
                    result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                else:
                    result_item["scores"][score_name] = float(score_values[i])

            results_this_rank.append(result_item)

        del images, all_scores
        torch.cuda.empty_cache()

    # --- Gather and Save Results ---
    dist.barrier()

    all_gathered_results = [None] * world_size
    dist.all_gather_object(all_gathered_results, results_this_rank)

    if is_main_process(rank):
        flat_results = [item for sublist in all_gathered_results for item in sublist]

        flat_results.sort(key=lambda x: x["sample_id"])

        with open(results_filepath, "w") as f_out:
            for result_item in flat_results:
                f_out.write(json.dumps(result_item) + "\n")

        print(f"\nEvaluation finished. All {len(flat_results)} results saved to {results_filepath}")

        all_scores_agg = defaultdict(list)

        for result in flat_results:
            for score_name, score_value in result["scores"].items():
                if isinstance(score_value, (int, float)):
                    all_scores_agg[score_name].append(score_value)

        average_scores = {
            name: np.mean(list(filter(lambda score: score != -10.0, scores))) for name, scores in all_scores_agg.items()
        }

        print("\n--- Average Scores ---")
        if not average_scores:
            print("No scores were found to average.")
        else:
            for name, avg_score in sorted(average_scores.items()):
                print(f"{name:<20}: {avg_score:.4f}")
        print("----------------------")

        avg_scores_filepath = os.path.join(args.output_dir, "average_scores.json")
        with open(avg_scores_filepath, "w") as f_avg:
            json.dump(average_scores, f_avg, indent=4)
        print(f"Average scores also saved to {avg_scores_filepath}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion model in a distributed manner.")
    parser.add_argument(
        "--lora_hf_path",
        type=str,
        default="",
        help="Huggingface path for LoRA.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
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
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench"], help="Dataset type."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_output",
        help="Directory to save evaluation results and generated images.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=40, help="Number of inference steps for the diffusion pipeline."
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the generated images.")
    parser.add_argument(
        "--save_images", action="store_true", help="Include this flag to save generated images to the output directory."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between 'no', 'fp16', or 'bf16'.",
    )

    args = parser.parse_args()
    main(args)
