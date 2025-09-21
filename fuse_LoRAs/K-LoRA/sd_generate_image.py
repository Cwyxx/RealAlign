import os
import argparse
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
from datasets import load_dataset, Dataset
from utils import insert_sd_klora_to_unet

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
parser.add_argument(
    "--lora_name_or_path_content",
    type=str,
    help="LoRA path",
    default="loraDataset/content_6/pytorch_lora_weights.safetensors",
)
parser.add_argument(
    "--lora_name_or_path_style",
    type=str,
    help="LoRA path",
    default="loraDataset/style_9/pytorch_lora_weights.safetensors",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save the generated images."
)
parser.add_argument(
    "--val_json_data_path",
    type=str,
    default="",
    help="Path to the JSON file containing validation data. This file is used during validation to generate images to evaluate model performance.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="A seed for reproducible training.",
)
parser.add_argument(
    "--image_column",
    type=str,
    default=None,
)
parser.add_argument(
    "--caption_column",
    type=str,
    default=None,
)
parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=50,
)
parser.add_argument(
    "--pattern",
    type=str,
    help="Pattern for the image generation",
    default="s*",
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

pattern = args.pattern
if pattern == "s*":
    alpha = 1.5
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = 0.5    
sum_timesteps = 28000

print(f"Content LoRA: {args.lora_name_or_path_content}")
print(f"Style LoRA: {args.lora_name_or_path_style}")
pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, safety_checker=None)
pipe.unet = insert_sd_klora_to_unet(
    pipe.unet, os.path.join(args.lora_name_or_path_content, "pytorch_lora_weights.safetensors"), os.path.join(args.lora_name_or_path_style, "pytorch_lora_weights.safetensors"), alpha, beta, sum_timesteps, pattern
)
pipe.to("cuda", dtype=torch.float16)

data_files = {}
data_files["val"] = args.val_json_data_path
dataset = load_dataset(
    os.path.splitext(args.val_json_data_path)[1][1:],
    data_files=data_files
)
val_dataset = dataset['val']
val_dataset = list(val_dataset)
val_dataset = sorted(val_dataset, key=lambda x: x[args.image_column])
val_dataset = val_dataset[:10_000]
val_dataset = Dataset.from_list(val_dataset)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=args.batch_size
)

pipe.set_progress_bar_config(
    leave=False,
    desc="Sampling Timestep",
    dynamic_ncols=True,
)

with torch.inference_mode():
    for batch in tqdm(val_dataloader):
        image_uids = batch[args.image_column]
        prompts = batch[args.caption_column]
        if args.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            generate_images = pipe(prompts, generator=[ generator ] * len(prompts), num_inference_steps=args.num_inference_steps).images
        else:
            generate_images = pipe(prompts, num_inference_steps=args.num_inference_steps).images
        for image_uid, generate_image in zip(image_uids, generate_images):
            generate_image.save(os.path.join(args.output_dir, f"{image_uid}.png"))