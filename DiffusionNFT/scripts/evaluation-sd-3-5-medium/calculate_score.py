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
sys.path.append(os.path.join(os.getcwd(), "../.."))
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import  AutoProcessor, AutoModelForCausalLM
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
    elif args.dataset == "x_aigd":
        dataset = TextPromptDataset(dataset_path, split="test")
    elif args.dataset == "pick_a_pic_v2":
        dataset = TextPromptDataset(dataset_path, split="test")
    elif args.dataset == "pickscore_validation":
        dataset = TextPromptDataset(dataset_path, split="validation")   
    elif args.dataset == "OneIG-Bench-Anime":
        dataset = TextPromptDataset(dataset_path, split="test")
    elif args.dataset == "OneIG-Bench-Portrait":
        dataset = TextPromptDataset(dataset_path, split="test")
        
    eval_batch_size = 2
    if args.reward_model in  ["hpsv3", "unifiedreward"]: eval_batch_size=1
    
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # --- Instantiate Reward Models ---
    print("Initializing reward models...")
    if args.reward_model in ["imagereward", "pickscore", "aesthetic", "clipscore", "hpsv2", "unifiedreward", "code", "dinov2"]:
        all_reward_scorers = { args.reward_model: 1.0 }
        scoring_fn, reward_models = multi_score(device, all_reward_scorers)
        for reward_model in reward_models.values(): reward_model.to(device)
        print(f"Initializing reward models {args.reward_model} from DiffusionNFT...")
        
    elif args.reward_model == "SGP-PickScore":
        all_reward_scorers = { "pickscore": 1.0 }
        pickscore_scoring_fn, pickscore_reward_models = multi_score(device, all_reward_scorers)
        for reward_model in pickscore_reward_models.values(): reward_model.to(device)
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            realistic_prompt = [ "Realistic photo " + prompt for prompt in prompts ]
            cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
            realistic_rewards,_ = pickscore_scoring_fn(images, realistic_prompt, metadata, only_strict)
            cg_rendered_rewards,_ = pickscore_scoring_fn(images, cg_rendered_prompt, metadata, only_strict)
            
            score_list = [realistic_reward - cg_rendered_reward for realistic_reward, cg_rendered_reward in zip(realistic_rewards["pickscore"], cg_rendered_rewards["pickscore"])]
            score_details = { args.reward_model: score_list }
            return score_details, {}
        
    elif args.reward_model == "SGP-ImageReward":
        all_reward_scorers = { "imagereward": 1.0 }
        imagereward_scoring_fn, imagereward_reward_models = multi_score(device, all_reward_scorers)
        for reward_model in imagereward_reward_models.values(): reward_model.to(device)
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            realistic_prompt = [ "Realistic photo " + prompt for prompt in prompts ]
            cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
            realistic_rewards,_ = imagereward_scoring_fn(images, realistic_prompt, metadata, only_strict)
            cg_rendered_rewards,_ = imagereward_scoring_fn(images, cg_rendered_prompt, metadata, only_strict)
            
            score_list = [realistic_reward - cg_rendered_reward for realistic_reward, cg_rendered_reward in zip(realistic_rewards["imagereward"], cg_rendered_rewards["imagereward"])]
            score_details = { args.reward_model: score_list }
            return score_details, {}
        
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
        
    elif args.reward_model == "q-align":
        import pyiqa
        reward_model = pyiqa.create_metric("qalign", device=device)
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
        reward_model = AutoModelForCausalLM.from_pretrained(
            "zhiyuanyou/DeQA-Score-Mix3",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
            revision="f37ba4273ad8d7548e21ac2fa58353c517e4df49"
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
        
    elif args.reward_model == "hpsv3":
        from hpsv3 import HPSv3RewardInferencer
        inferencer = HPSv3RewardInferencer(device='cuda')
        # inferencer.model = inferencer.model.to(device).to(torch.float16) 
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            assert type(images[0]) == str
            image_paths = images
            with torch.no_grad():
                rewards = inferencer.reward(prompts=prompts, image_paths=image_paths)
                score_list = [reward[0].item() for reward in rewards]
            
            score_details = { args.reward_model: score_list }
            return score_details, {}
        
    elif args.reward_model == "SGP-HPSv3":
        from hpsv3 import HPSv3RewardInferencer
        inferencer = HPSv3RewardInferencer(device='cuda')
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            assert type(images[0]) == str
            image_paths = images
            realistic_prompt = [ "Realistic photo " + prompt for prompt in prompts ]
            cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
            with torch.no_grad():
                realistic_rewards = inferencer.reward(prompts=realistic_prompt, image_paths=image_paths)
                cg_rendered_rewards = inferencer.reward(prompts=cg_rendered_prompt, image_paths=image_paths)
                score_list = [realistic_reward[0].item() - cg_rendered_reward[0].item() for realistic_reward, cg_rendered_reward in zip(realistic_rewards, cg_rendered_rewards)]
            
            score_details = { args.reward_model: score_list }
            return score_details, {}
        
    elif args.reward_model == "cpbd":
        import cpbd

        def scoring_fn(images, prompts, metadata, only_strict=False):
            score_list = []
            image_paths = images
            for image_path in image_paths:
                img = np.array(Image.open(image_path).convert('L'))
                score = cpbd.compute(img)
                score_list.append(score)
            
            score_details = { args.reward_model: score_list }
            return score_details, {}
        
    elif args.reward_model == "imagedoctor":
        from imagedoctor import ImageDoctor
        reward_model = ImageDoctor(image_dir=args.output_dir)
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            score_list = []
            image_paths = images
            
            for image_path, prompt in zip(image_paths, prompts):
                perceptual_artifact_ratio = reward_model(image_path, prompt)
                score_list.append(perceptual_artifact_ratio)
                
            score_details = { args.reward_model: score_list }
            return score_details, {}
        
    elif args.reward_model == "diffdoctor":
        from diffdoctor import DiffDoctor
        reward_model = DiffDoctor(image_dir=args.output_dir)
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            score_list = []
            image_paths = images
            
            for image_path in image_paths:
                perceptual_artifact_ratio = reward_model(image_path)
                score_list.append(perceptual_artifact_ratio)
                
            score_details = { args.reward_model: score_list }
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
        
        if args.reward_model in [ "imagereward",  "pickscore",  "aesthetic", "clipscore", "hpsv2", "unifiedreward", "code", "dinov2", "SGP-PickScore", "SGP-ImageReward"]:
            pil_images = [ Image.open(image_path) for image_path in image_paths ] # imagereward, pickscore get pil image input
            
            if args.reward_model == "aesthetic":
                pil_images = np.stack([np.array(img) for img in pil_images])
                images = pil_images.transpose(0, 3, 1, 2)
                images = torch.tensor(images, dtype=torch.uint8)
                
            if args.reward_model in [ "clipscore", "hpsv2"]:
                images = [ np.array(img) for img in pil_images ]
                images = np.array(images)
                images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
                images = torch.tensor(images, dtype=torch.uint8) / 255.0
            
            elif args.reward_model in [ "imagereward", "pickscore", "unifiedreward", "code", "dinov2", "SGP-PickScore", "SGP-ImageReward"]:
                images = pil_images
        
        elif args.reward_model in [ "vqascore", "clip_iqa", "deqa", "aesthetic_v2_5", "hpsv3", "cpbd", "q-align", "imagedoctor", "diffdoctor", "SGP-HPSv3" ]:
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

        torch.cuda.empty_cache()

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
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench", "pick_a_pic_spo", "drawbench-analysis", "x_aigd", "pick_a_pic_v2", "pickscore_validation", "drawbench_realistic_style", "OneIG-Bench-Anime", "OneIG-Bench-Portrait"], help="Dataset type."
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
