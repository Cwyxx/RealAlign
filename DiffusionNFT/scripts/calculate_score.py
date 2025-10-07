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
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from flow_grpo.rewards import multi_score
from collections import defaultdict
import random


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]
            
        random.seed(42)
        self.prompts = random.sample(self.prompts, 5)
        print(f"Sampled 5 prompts for quick testing:\n{self.prompts}")

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
    dataset_path = f"../dataset/{args.dataset}"
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
    eval_batch_size = 2
    
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
        scoring_fn, reward_models = multi_score(device, all_reward_scorers)
        for reward_model in reward_models.values(): reward_model.to(device)
        print(f"Initializing reward models {args.reward_model} from DiffusionNFT...")
        
    elif args.reward_model == "vqascore":
        import t2v_metrics
        clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl')
        print(f"Initializing reward models {args.reward_model}...")
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            scores = clip_flant5_score(images=images, texts=prompts)
            diagonal_scores = [scores[i][i] for i in range(len(scores))]
            score_details = { args.reward_model: diagonal_scores}
            return score_details, {}
        
    elif args.reward_model == "clip_iqa":
        import pyiqa
        reward_model = pyiqa.create_metric("clipiqa", device=device)
        print(f"Initializing reward models {args.reward_model}...")
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            score_list = []
            for image in images:
                score = reward_model(image)
                if isinstance(score, torch.Tensor):
                    score = score.item()
                score_list.append(score)
                
            score_details = { args.reward_model: score_list}
            return score_details, {}
        
    elif args.reward_model == "deqa":
        from transformers import AutoModelForCausalLM
        reward_model = AutoModelForCausalLM.from_pretrained(
            "zhiyuanyou/DeQA-Score-Mix3",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"Initializing reward models {args.reward_model}...")
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            images = [ Image.open(image_path) for image_path in images ]
            score_list = reward_model.score(images).tolist()
            
            score_details = { args.reward_model: score_list}
            return score_details, {}
        
    elif args.reward_model == "aesthetic_v2_5":
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        reward_model, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        reward_model = reward_model.to(device)
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            score_list = []
            for image_path in images:
                pil_image = Image.open(image_path).convert("RGB")
                pixel_values = (
                    preprocessor(images=pil_image, return_tensors="pt")
                    .pixel_values.to(device)
                )
                score = reward_model(pixel_values).logits.squeeze().float().detach().cpu().numpy()
                score_list.append(score.item())
                
            score_details = { args.reward_model: score_list}
            return score_details, {}

    score_list = []
    # --- Evaluation Loop ---
    result_this_rank = []
    with open(results_filepath, 'r') as f:
        for line in f:
            if line.strip(): result_this_rank.append(json.loads(line)) # Skip any empty lines
    result_this_rank.sort(key=lambda x: x["sample_id"])

    for batch in tqdm(dataloader, desc=f"Evaluating"):
        prompts, metadata, indices = batch
        current_batch_size = len(prompts)
        image_paths = [ os.path.join(args.output_dir, "images", f"{sample_idx:05d}.png") for sample_idx in indices ]
        
        if args.reward_model in [ "imagereward",  "pickscore",  "aesthetic", "clipscore", "hpsv2"]:
            pil_images = [ Image.open(image_path) for image_path in image_paths ] # imagereward, pickscore get pil image input
            
            if args.reward_model == "aesthetic":
                pil_images = np.stack([np.array(img) for img in pil_images])
                images = pil_images.transpose(0, 3, 1, 2)
                images = torch.tensor(images, dtype=torch.uint8)
                
            elif args.reward_model in [ "clipscore", "hpsv2"]:
                images = [ np.array(img) for img in pil_images ]
                images = np.array(images)
                images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
                images = torch.tensor(images, dtype=torch.uint8) / 255.0
            
            elif args.reward_model in [ "imagereward", "pickscore"]:
                images = pil_images
        
        elif args.reward_model in [ "vqascore", "clip_iqa", "deqa", "aesthetic_v2_5" ]:
            images = image_paths # path
                
        all_scores, _ = scoring_fn(images, prompts, metadata, only_strict=False) # calculate_score
        
        for i in range(current_batch_size):
            sample_idx = indices[i]
            result_item = result_this_rank[sample_idx]
            assert result_item["sample_id"] == sample_idx, f'result_item[sample_id]: {result_item["sample_id"]} / sample_idx: {sample_idx}'
            
            for score_name, score_values in all_scores.items():
                if isinstance(score_values, torch.Tensor):
                    result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                else:
                    result_item["scores"][score_name] = float(score_values[i])
                    score_list.append(float(score_values[i]))

        del images, all_scores
        torch.cuda.empty_cache()

    print(f"score_list / {args.reward_model}: {sum(score_list)/len(score_list):.4f} / len: {len(score_list)}")
    result_this_rank.sort(key=lambda x: x["sample_id"])

    with open(results_filepath, "w") as f_out:
        for result_item in result_this_rank:
            f_out.write(json.dumps(result_item) + "\n")

    print(f"\nEvaluation finished. All {len(result_this_rank)} results saved to {results_filepath}")

    all_scores_agg = defaultdict(list)

    for result in result_this_rank:
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
            print(f"{name:<20}: {avg_score:.10f}")
    print("----------------------")

    avg_scores_filepath = os.path.join(args.output_dir, "average_scores.json")
    with open(avg_scores_filepath, "w") as f_avg:
        json.dump(average_scores, f_avg, indent=4)
    print(f"Average scores also saved to {avg_scores_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion model in a distributed manner.")
    parser.add_argument(
        "--reward_model",
        type=str,
        default=None,
        help="Reward Model.",
    )
    parser.add_argument(
        "--lora_hf_path",
        type=str,
        default=None,
        help="Huggingface path for LoRA.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Local path to the LoRA checkpoint directory (e.g., './save/run_name/checkpoints/checkpoint-5000').",
    )
    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     required=True,
    #     choices=["sd3"],
    #     help="Type of the base model ('sd3').",
    # )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench", "pick_a_pic_spo", "drawbench-analysis"], help="Dataset type."
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
