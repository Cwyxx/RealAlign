import os
import argparse
from tqdm import tqdm
import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from peft import get_peft_model, LoraConfig
import copy
from datasets import load_dataset, Dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help=(
            "Path to a pretrained Stable Diffusion model or its identifier from Hugging Face Hub. "
            "This is used to load the base model for training or inference. "
            "Default is 'CompVis/stable-diffusion-v1-4'."
        )
    )
    
    ### Diffusers to PEFT ###
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--peft_dir",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.peft_dir, exist_ok=True)
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        subfolder="unet",
    ).to("cuda")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        unet=unet
    ).to("cuda")
    pipeline.load_lora_weights(args.lora_dir, weight_name="pytorch_lora_weights.safetensors", adapter_name=args.adapter_name)
    
    
    sd_unet = copy.deepcopy(unet)
    adapter_peft_model = get_peft_model(
        sd_unet,
        pipeline.unet.peft_config[args.adapter_name],
        adapter_name=args.adapter_name
    )
    
    original_state_dict = {f"base_model.model.{k}": v for k, v in pipeline.unet.state_dict().items()}
    adapter_peft_model.load_state_dict(original_state_dict, strict=True)
    adapter_peft_model.save_pretrained(args.peft_dir)
    print(f"Loaded LoRA from {args.lora_dir} with adapter {args.adapter_name}")
    print(f"Saved PEFT model to {args.peft_dir}")