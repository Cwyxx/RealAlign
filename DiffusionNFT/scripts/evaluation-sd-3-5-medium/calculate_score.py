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
    if args.reward_model in ["shannon_entropy", "GLCM_homogeneity", "GLCM_contrast", "laplacian_variance", "LBP", "edge_density" ]:
        from skimage.measure import shannon_entropy
        from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
        from skimage import io, color, img_as_ubyte
        import cv2
        from scipy.stats import entropy
        
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
    elif args.dataset == "drawbench-unique":
        dataset = TextPromptDataset(dataset_path, split="test")
    elif args.dataset == "HPDv2-all":
        dataset = TextPromptDataset(dataset_path, split="test")
    elif args.dataset == "partiprompts":
        dataset = TextPromptDataset(dataset_path, split="test")
        
    eval_batch_size = 2
    if args.reward_model in  ["hpsv3", "unifiedreward", "unifiedreward_2", "color-fidelity-metric"]: eval_batch_size=1
    
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # --- Instantiate Reward Models ---
    print("Initializing reward models...")
    if args.reward_model in ["imagereward", "pickscore", "aesthetic", "clipscore", "hpsv2", "unifiedreward", "code", "dinov2", "unifiedreward_2"]:
        all_reward_scorers = { args.reward_model: 1.0 }
        scoring_fn, reward_models = multi_score(device, all_reward_scorers)
        for reward_model in reward_models.values(): reward_model.to(device)
        print(f"Initializing reward models {args.reward_model} from DiffusionNFT...")
        
    elif args.reward_model == "color-fidelity-metric":
        from CFM.cfm.inference import CFMRewardInferencer
        
        cfm_model = CFMRewardInferencer()
        
        def scoring_fn(images, prompts, metadata, only_strict=False):
            ### images is image_paths #### 
            assert type(images[0]) == str
            image_paths = images
            score_list = cfm_model.reward(prompts, image_paths)
            score_details = { args.reward_model: score_list}
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
        reward_model = reward_model.to(device)
        
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
        
        if args.reward_model in [ "imagereward",  "purple-pickscore", "purple-hpsv3", "pickscore",  "aesthetic", "clipscore", "hpsv2", "unifiedreward", "code", "dinov2" ]:
            pil_images = [ Image.open(image_path) for image_path in image_paths ] # imagereward, pickscore get pil image input
            
            if args.reward_model == "aesthetic":
                pil_images = np.stack([np.array(img) for img in pil_images])
                images = pil_images.transpose(0, 3, 1, 2)
                images = torch.tensor(images, dtype=torch.uint8)
                
            if args.reward_model in [ "SGP-v2-clipscore", "clipscore", "hpsv2"]:
                images = [ np.array(img) for img in pil_images ]
                images = np.array(images)
                images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
                images = torch.tensor(images, dtype=torch.uint8) / 255.0
            
            elif args.reward_model in [ "imagereward", "purple-pickscore", "pickscore", "unifiedreward", "code", "dinov2" ]:
                images = pil_images
        
        elif args.reward_model in [ "deqa", "aesthetic_v2_5", "hpsv3", "cpbd", "q-align", "imagedoctor", "diffdoctor", "color-fidelity-metric" ]:
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
        "--dataset", type=str, required=True, choices=["partiprompts", "geneval", "ocr", "pickscore", "drawbench", "pick_a_pic_spo", "drawbench-analysis", "x_aigd", "pick_a_pic_v2", "pickscore_validation", "drawbench_realistic_style", "OneIG-Bench-Anime", "OneIG-Bench-Portrait", "drawbench-unique", "HPDv2-anime", "HPDv2-concept-art", "HPDv2-paintings", "HPDv2-photo", "HPDv2-photo-all", "HPDv2-anime-all", "HPDv2-all"], help="Dataset type."
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

    # elif args.reward_model == "SGP-PickScore":
    #     all_reward_scorers = { "pickscore": 1.0 }
    #     pickscore_scoring_fn, pickscore_reward_models = multi_score(device, all_reward_scorers)
    #     for reward_model in pickscore_reward_models.values(): reward_model.to(device)
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         realistic_prompt = [ "Realistic photo " + prompt for prompt in prompts ]
    #         cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
    #         realistic_rewards,_ = pickscore_scoring_fn(images, realistic_prompt, metadata, only_strict)
    #         cg_rendered_rewards,_ = pickscore_scoring_fn(images, cg_rendered_prompt, metadata, only_strict)
            
    #         score_list = [realistic_reward - cg_rendered_reward for realistic_reward, cg_rendered_reward in zip(realistic_rewards["pickscore"], cg_rendered_rewards["pickscore"])]
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "purple-pickscore":
    #     all_reward_scorers = { "pickscore": 1.0 }
    #     pickscore_scoring_fn, pickscore_reward_models = multi_score(device, all_reward_scorers)
    #     for reward_model in pickscore_reward_models.values(): reward_model.to(device)
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         normal_prompt = [ prompt for prompt in prompts ]
    #         purple_prompt = [ "purple color tones; " + prompt for prompt in prompts ]
    #         normal_rewards,_ = pickscore_scoring_fn(images, normal_prompt, metadata, only_strict)
    #         purple_rewards,_ = pickscore_scoring_fn(images, purple_prompt, metadata, only_strict)
            
    #         score_list = [ normal_reward - purple_reward for normal_reward, purple_reward in zip(normal_rewards["pickscore"], purple_rewards["pickscore"])]
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "SGP-v2-clipscore":
    #     all_reward_scorers = { "clipscore": 1.0 }
    #     clipscore_scoring_fn, clipscore_reward_models = multi_score(device, all_reward_scorers)
    #     for reward_model in clipscore_reward_models.values(): reward_model.to(device)
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         normal_prompt = [ prompt for prompt in prompts ]
    #         cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
    #         normal_rewards,_ = clipscore_scoring_fn(images, normal_prompt, metadata, only_strict)
    #         cg_rendered_rewards,_ = clipscore_scoring_fn(images, cg_rendered_prompt, metadata, only_strict)
    #         score_list = [cg_rendered_reward - normal_reward for normal_reward, cg_rendered_reward in zip(normal_rewards["clipscore"], cg_rendered_rewards["clipscore"])]
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "SGP-ImageReward":
    #     all_reward_scorers = { "imagereward": 1.0 }
    #     imagereward_scoring_fn, imagereward_reward_models = multi_score(device, all_reward_scorers)
    #     for reward_model in imagereward_reward_models.values(): reward_model.to(device)
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         realistic_prompt = [ "Realistic photo " + prompt for prompt in prompts ]
    #         cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
    #         realistic_rewards,_ = imagereward_scoring_fn(images, realistic_prompt, metadata, only_strict)
    #         cg_rendered_rewards,_ = imagereward_scoring_fn(images, cg_rendered_prompt, metadata, only_strict)
            
    #         score_list = [realistic_reward - cg_rendered_reward for realistic_reward, cg_rendered_reward in zip(realistic_rewards["imagereward"], cg_rendered_rewards["imagereward"])]
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
    
    # elif args.reward_model == "clip_iqa":
    #     import pyiqa
    #     reward_model = pyiqa.create_metric("clipiqa", device=device)
    #     print(f"Initializing reward models {args.reward_model}...")
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         score_list = []
    #         for image in images:
    #             score = reward_model(image)
    #             if isinstance(score, torch.Tensor):
    #                 score = score.item()
    #             score_list.append(score)
                
    #         score_details = { args.reward_model: score_list}
    #         return score_details, {}
        
    # elif args.reward_model == "clip_iqa":
    #     import pyiqa
    #     reward_model = pyiqa.create_metric("n", device=device)
    #     print(f"Initializing reward models {args.reward_model}...")
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         score_list = []
    #         for image in images:
    #             score = reward_model(image)
    #             if isinstance(score, torch.Tensor):
    #                 score = score.item()
    #             score_list.append(score)
                
    #         score_details = { args.reward_model: score_list}
    #         return score_details, {}
        
    # elif args.reward_model == "niqe":
    #     import pyiqa
    #     reward_model = pyiqa.create_metric("niqe", device=device)
    #     print(f"Initializing reward models {args.reward_model}...")
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         score_list = []
    #         for image in images:
    #             score = reward_model(image)
    #             if isinstance(score, torch.Tensor):
    #                 score = score.item()
    #             score_list.append(score)
                
    #         score_details = { args.reward_model: score_list}
    #         return score_details, {}
        
    # elif args.reward_model == "brisque":
    #     import pyiqa
    #     reward_model = pyiqa.create_metric("brisque", device=device)
    #     print(f"Initializing reward models {args.reward_model}...")
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         score_list = []
    #         for image in images:
    #             score = reward_model(image)
    #             if isinstance(score, torch.Tensor):
    #                 score = score.item()
    #             score_list.append(score)
                
    #         score_details = { args.reward_model: score_list}
    #         return score_details, {}
        
    # elif args.reward_model == "q-align":
    #     import pyiqa
    #     reward_model = pyiqa.create_metric("qalign", device=device)
    #     print(f"Initializing reward models {args.reward_model}...")
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         score_list = []
    #         for image in images:
    #             score = reward_model(image)
    #             if isinstance(score, torch.Tensor):
    #                 score = score.item()
    #             score_list.append(score)
                
    #         score_details = { args.reward_model: score_list}
    #         return score_details, {}
    
            
    # elif args.reward_model == "GLCM_homogeneity":
    
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         score_list = []
    #         image_paths = images
            
    #         for image_path in image_paths:
    #             # 1. 加载图像并转换为灰度图
    #             # 真实场景中，AI生成的图通常是RGB，需转为灰度以计算纹理指标
    #             img = io.imread(image_path)
    #             if len(img.shape) == 3:
    #                 gray_img = color.rgb2gray(img)
    #             else:
    #                 gray_img = img

    #             # 2. 将图像转换为 8-bit 无符号整型 (0-255)
    #             # GLCM 要求离散灰度级，这是计算的前提
    #             gray_img_8bit = img_as_ubyte(gray_img)

    #             # 3. 计算 GLCM
    #             # distances: 像素对的距离（通常选 1-5 像素）
    #             # angles: 方向（0°, 45°, 90°, 135°）。通常取平均值以获得旋转不变性
    #             glcm = graycomatrix(gray_img_8bit, 
    #                                 distances=[1], 
    #                                 angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
    #                                 levels=256, 
    #                                 symmetric=True, 
    #                                 normed=True)

    #             # 4. 计算 Homogeneity
    #             homogeneity_values = graycoprops(glcm, 'homogeneity')
                
    #             score_list.append(np.mean(homogeneity_values).item())
                
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "GLCM_contrast":
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         score_list = []
    #         image_paths = images
            
    #         for image_path in image_paths:
    #             # 1. 加载图像并转换为灰度图
    #             # 真实场景中，AI生成的图通常是RGB，需转为灰度以计算纹理指标
    #             img = io.imread(image_path)
    #             if len(img.shape) == 3:
    #                 gray_img = color.rgb2gray(img)
    #             else:
    #                 gray_img = img

    #             # 2. 将图像转换为 8-bit 无符号整型 (0-255)
    #             # GLCM 要求离散灰度级，这是计算的前提
    #             gray_img_8bit = img_as_ubyte(gray_img)

    #             # 3. 计算 GLCM
    #             # distances: 像素对的距离（通常选 1-5 像素）
    #             # angles: 方向（0°, 45°, 90°, 135°）。通常取平均值以获得旋转不变性
    #             glcm = graycomatrix(gray_img_8bit, 
    #                                 distances=[1], 
    #                                 angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
    #                                 levels=256, 
    #                                 symmetric=True, 
    #                                 normed=True)

    #             # 4. 计算 Homogeneity
    #             contrast_values = graycoprops(glcm, 'contrast')
                
    #             score_list.append(np.mean(contrast_values).item())
                
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
    
    # elif args.reward_model == "shannon_entropy":
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         score_list = []
    #         image_paths = images
            
    #         for image_path in image_paths:
    #             img = io.imread(image_path)
    #             # 转换为灰度图计算纹理信息量
    #             if len(img.shape) == 3:
    #                 gray_img = color.rgb2gray(img)
    #             else:
    #                 gray_img = img
                
    #             # 将图像转为 8-bit (0-255)
    #             gray_img_8bit = img_as_ubyte(gray_img)
                
    #             # 计算香农熵
    #             entropy_value = shannon_entropy(gray_img_8bit)
    #             score_list.append(entropy_value)
                
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
    
    # elif args.reward_model == "laplacian_variance":
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         score_list = []
    #         image_paths = images
            
    #         for image_path in image_paths:
    #             # 1. 读取图像
    #             img = io.imread(image_path)
                
    #             # 2. 转为灰度图 (必须)
    #             if len(img.shape) == 3:
    #                 gray = color.rgb2gray(img)
    #             else:
    #                 gray = img
                    
    #             # 3. 转为 uint8 格式 (OpenCV通常处理0-255)
    #             gray = img_as_ubyte(gray)
                
    #             # 4. 计算拉普拉斯算子
    #             # ddepth=cv2.CV_64F 防止溢出
    #             laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                
    #             # 5. 计算方差
    #             variance = laplacian.var()
    #             score_list.append(variance)
                
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "LBP":    
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         score_list = []
    #         image_paths = images
    #         radius = 1
    #         n_points = 8
    #         for image_path in image_paths:
    #             # 1. 读取图像并转换为灰度
    #             img = io.imread(image_path)
    #             if len(img.shape) == 3:
    #                 gray = color.rgb2gray(img)
    #             else:
    #                 gray = img
                
    #             # 转换为 8-bit 整数 (0-255)
    #             gray = img_as_ubyte(gray)
                
    #             # 2. 计算 LBP 特征图
    #             # 'uniform' 模式具有旋转不变性，且能有效减少特征维度，是分析纹理的标准选择
    #             lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
                
    #             # 3. 计算 LBP 直方图 (概率分布)
    #             # uniform 模式下的 bin 数量为 n_points + 2
    #             n_bins = int(lbp.max() + 1)
    #             hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                
    #             # 4. 计算香农熵 (Shannon Entropy)
    #             # 熵越高 = 纹理模式越多样 = 细节越丰富
    #             # 使用 base=2，单位为 bit
    #             lbp_entropy_score = entropy(hist, base=2)
                
    #             score_list.append(lbp_entropy_score)
                
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "edge_density":    
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         score_list = []
    #         image_paths = images
    #         low_threshold = 100
    #         high_threshold = 200
    #         for image_path in image_paths:
    #             # 1. 读取并转灰度
    #             img = io.imread(image_path)
    #             if len(img.shape) == 3:
    #                 gray = color.rgb2gray(img)
    #             else:
    #                 gray = img
    #             gray = img_as_ubyte(gray)
                
    #             # 2. Canny 边缘检测
    #             # 阈值选取很关键，100/200 是比较标准的自然图像设置
    #             edges = cv2.Canny(gray, low_threshold, high_threshold)
                
    #             # 3. 计算边缘密度 (非零像素占比)
    #             # 结果是百分比，比如 0.05 表示 5% 的像素是边缘
    #             density = np.count_nonzero(edges) / edges.size
                
    #             score_list.append(density)
                
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "saturation":
    #     def calculate_saturation(image_path):
    #         img = Image.open(image_path).convert("HSV")
    #         s_channel = np.array(img)[:, :, 1]
    #         return s_channel.mean()
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         #### images is image_paths ####
    #         image_paths = images
    #         score_list = [calculate_saturation(image_path) for image_path in image_paths]
    #         score_details = { "saturation": score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "brightness":
    #     def calculate_average_brightness(image_path):
    #         image = Image.open(image_path)
    #         gray_image = image.convert('L')
    #         gray_array = np.array(gray_image)
    #         return np.mean(gray_array)
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         #### images is image_paths ####
    #         image_paths = images
    #         score_list = [calculate_average_brightness(image_path) for image_path in image_paths]
    #         score_details = { "brightness": score_list }
    #         return score_details, {}


    # elif args.reward_model == "purple-hpsv3":
    #     from hpsv3 import HPSv3RewardInferencer
    #     inferencer = HPSv3RewardInferencer(device='cuda')
    #     # inferencer.model = inferencer.model.to(device).to(torch.float16) 
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         assert type(images[0]) == str
    #         image_paths = images
    #         purple_prompt = [ "purple color tones; " + prompt for prompt in prompts ]
    #         normal_prompt = [ prompt for prompt in prompts ]
    #         with torch.no_grad():
    #             normal_rewards = inferencer.reward(prompts=normal_prompt, image_paths=image_paths)
    #             purple_rewards = inferencer.reward(prompts=purple_prompt, image_paths=image_paths)
    #             score_list = [normal_reward[0].item() - purple_reward[0].item() for normal_reward, purple_reward in zip(normal_rewards, purple_rewards)]
            
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "SGP-HPSv3":
    #     from hpsv3 import HPSv3RewardInferencer
    #     inferencer = HPSv3RewardInferencer(device='cuda')
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         assert type(images[0]) == str
    #         image_paths = images
    #         realistic_prompt = [ "Realistic photo " + prompt for prompt in prompts ]
    #         cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
    #         with torch.no_grad():
    #             realistic_rewards = inferencer.reward(prompts=realistic_prompt, image_paths=image_paths)
    #             cg_rendered_rewards = inferencer.reward(prompts=cg_rendered_prompt, image_paths=image_paths)
    #             score_list = [realistic_reward[0].item() - cg_rendered_reward[0].item() for realistic_reward, cg_rendered_reward in zip(realistic_rewards, cg_rendered_rewards)]
            
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "SGP-v2-HPSv3":
    #     from hpsv3 import HPSv3RewardInferencer
    #     inferencer = HPSv3RewardInferencer(device='cuda')
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         assert type(images[0]) == str
    #         image_paths = images
    #         cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
    #         normal_prompt = [ prompt for prompt in prompts ]
    #         with torch.no_grad():
    #             cg_rendered_rewards = inferencer.reward(prompts=cg_rendered_prompt, image_paths=image_paths)
    #             normal_rewards = inferencer.reward(prompts=normal_prompt, image_paths=image_paths)
    #             score_list = [cg_rendered_reward[0].item() - normal_reward[0].item() for normal_reward, cg_rendered_reward in zip(normal_rewards, cg_rendered_rewards)]
            
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "SGP-v1.5-HPSv3":
    #     from hpsv3 import HPSv3RewardInferencer
    #     inferencer = HPSv3RewardInferencer(device='cuda')
        
    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         ### images is image_paths #### 
    #         assert type(images[0]) == str
    #         image_paths = images
    #         cg_rendered_prompt = [ "CG Render " + prompt for prompt in prompts ]
    #         realistic_prompt = [ "Realistic photo " + prompt for prompt in prompts ]
    #         with torch.no_grad():
    #             cg_rendered_rewards = inferencer.reward(prompts=cg_rendered_prompt, image_paths=image_paths)
    #             realistic_rewards = inferencer.reward(prompts=realistic_prompt, image_paths=image_paths)
    #             score_list = [ cg_rendered_reward[0].item() - realistic_reward[0].item() for realistic_reward, cg_rendered_reward in zip(realistic_rewards, cg_rendered_rewards)]
            
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        
    # elif args.reward_model == "cpbd":
    #     import cpbd

    #     def scoring_fn(images, prompts, metadata, only_strict=False):
    #         score_list = []
    #         image_paths = images
    #         for image_path in image_paths:
    #             img = np.array(Image.open(image_path).convert('L'))
    #             score = cpbd.compute(img)
    #             score_list.append(score)
            
    #         score_details = { args.reward_model: score_list }
    #         return score_details, {}
        