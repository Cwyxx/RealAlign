import os
import argparse
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
from datasets import load_dataset, Dataset

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
    "--lora_weight_dir",
    type=str,
    default=None,
    help=(
        "Path to the directory containing LoRA weights to be loaded. "
        "If specified, the model will load the LoRA weights from this directory for fine-tuned inference or training. "
        "Default is None, meaning no LoRA weights are loaded."
    )
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
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker=None).to('cuda')
print(f"**************************************************")
if os.path.exists(args.lora_weight_dir):
    print(f"loading lora weights from {args.lora_weight_dir}")
    pipeline.load_lora_weights(
        args.lora_weight_dir,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="default"
    )
    pipeline.set_adapters("default")
else:
    print(f"{args.pretrained_model_name_or_path}")

print(f"output image dir: {args.output_dir}")
print(f"Activate Lora: {pipeline.get_active_adapters()}")
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

pipeline.set_progress_bar_config(
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
            generate_images = pipeline(prompts, generator=[ generator ] * len(prompts)).images
        else:
            generate_images = pipeline(prompts).images
        for image_uid, generate_image in zip(image_uids, generate_images):
            generate_image.save(os.path.join(args.output_dir, f"{image_uid}.png"))