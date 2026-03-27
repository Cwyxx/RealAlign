from PIL import Image
import torch
import re
import base64
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import argparse
import json

def extract_scores(output_logits, processor):
    vocab = processor.tokenizer.get_vocab()
    probabilities = output_logits[0, -1, :]
    # probabilities = F.softmax(last_token_logits, dim=-1)
    probabilities_ = probabilities.float().cpu().numpy()
    # fake_score = max(probabilities_[vocab['fake']], probabilities_[vocab['Fake']])
    # real_score = max(probabilities_[vocab['Real']], probabilities_[vocab['real']])
    fake_score = (probabilities_[vocab['fake']] + probabilities_[vocab['Fake']])/2
    real_score = (probabilities_[vocab['Real']] + probabilities_[vocab['real']])/2
    compare_score = np.array([fake_score, real_score])
    e_x = np.exp(compare_score - np.max(compare_score))
    score = e_x / e_x.sum()
    return score[1]

class QwenVLScorer(torch.nn.Module):
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            # attn_implementation="flash_attention_2",
            device_map=None,
        ).to(self.device)
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        
    @torch.no_grad()
    def __call__(self, images):
        rewards = []
        for base64_qwen in tqdm(images):
            messages=[]
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_qwen},
                        {"type": "text", "text": (
                            "Analyze the provided image. "
                            "Decide whether it is a real photograph or AI-generated. "
                            "The first word must be either 'real' or 'fake'."
                        )},
                    ],
                },
            ])

            # Preparation for batch inference
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits

            reward = extract_scores(logits, self.processor)
            rewards.append(reward)
        return rewards
    

# Usage example
def main():
    parser = argparse.ArgumentParser(description="Forensic-Chat evaluation for RealGen benchmark")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/model-ckpt/Forensic-Chat/Forensic-Chat",
        help="Path to Forensic-Chat model"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing generated images (expects images/ subdirectory)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save evaluation results (JSON format)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode"
    )
    args = parser.parse_args()

    # Set dtype based on mixed precision
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"Loading Forensic-Chat model from: {args.model_path}")
    scorer = QwenVLScorer(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype
    )

    # Load images from directory
    images_dir = os.path.join(args.image_dir, "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"Found {len(image_paths)} images in {images_dir}")

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_dir}")

    # Run evaluation
    print("Running Forensic-Chat evaluation...")
    result = scorer(image_paths)
    mean_score = sum(result) / len(result)

    # Save results
    results = {
        "model_path": args.model_path,
        "image_dir": args.image_dir,
        "num_images": len(image_paths),
        "mean_score": float(mean_score),
        "scores": [float(s) for s in result]
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nMean score: {mean_score:.4f}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()