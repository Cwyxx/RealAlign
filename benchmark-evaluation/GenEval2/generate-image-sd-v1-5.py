import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torch.utils.data import DataLoader, Dataset


class GenEval2PromptDataset(Dataset):
    """GenEval2 benchmark: load prompts from a JSONL file."""

    def __init__(self, benchmark_data_path):
        if not os.path.isfile(benchmark_data_path):
            raise FileNotFoundError(f"GenEval2 data file not found: {benchmark_data_path}")

        with open(benchmark_data_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        assert len(self.data) == 800, f"GenEval2 expects 800 prompts, got {len(self.data)}"
        print(f"Loaded {len(self.data)} prompts from {benchmark_data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "prompt": self.data[idx]["prompt"],
            "index": idx,
        }


def collate_fn(examples):
    prompts = [e["prompt"] for e in examples]
    indices = [e["index"] for e in examples]
    return prompts, indices


def main(args):
    device = torch.device("cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    mapping_filepath = os.path.join(args.output_dir, "prompt_image_mapping.json")

    # --- Load Model and Pipeline ---
    unet_init = args.unet_init
    print(f"Loading UNet from: {unet_init}")
    unet = UNet2DConditionModel.from_pretrained(unet_init, subfolder="unet")
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.unet = unet

    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        print(f"Loading LoRA weights from: {args.checkpoint_path}")
        pipeline.load_lora_weights(args.checkpoint_path, weight_name="pytorch_lora_weights.safetensors")
        print(f"Active adapters: {pipeline.get_active_adapters()}")

    pipeline.to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # --- Load Dataset ---
    benchmark_data_path = args.benchmark_data
    if not benchmark_data_path:
        benchmark_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geneval2_data.jsonl")

    dataset = GenEval2PromptDataset(benchmark_data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
    )

    prompt_image_mapping = {}

    for batch in tqdm(dataloader, desc="Generating images"):
        prompts, indices = batch
        generator = torch.Generator(device).manual_seed(args.seed)

        with torch.no_grad():
            images = pipeline(
                prompts,
                output_type="pt",
                height=args.resolution,
                width=args.resolution,
                generator=generator,
            )[0]

        for i, (prompt, idx) in enumerate(zip(prompts, indices)):
            image_path = os.path.join(images_dir, f"{idx:05d}.png")
            pil_image = Image.fromarray(
                (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil_image.save(image_path)
            prompt_image_mapping[prompt] = image_path

    with open(mapping_filepath, "w", encoding="utf-8") as f:
        json.dump(prompt_image_mapping, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(prompt_image_mapping)} images saved to: {images_dir}")
    print(f"Prompt-image mapping saved to: {mapping_filepath}")
    print(f"\nNext step — run evaluation:")
    print(
        f"  python evaluation.py \\\n"
        f"      --benchmark_data {benchmark_data_path} \\\n"
        f"      --image_filepath_data {mapping_filepath} \\\n"
        f"      --method soft_tifa_gm \\\n"
        f"      --output_file {os.path.join(args.output_dir, 'scores.json')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images for GenEval2 benchmark using SD v1-5.")
    parser.add_argument(
        "--benchmark_data",
        type=str,
        default=None,
        help="Path to geneval2_data.jsonl. Default: <script_dir>/geneval2_data.jsonl.",
    )
    parser.add_argument(
        "--unet_init",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        choices=[
            "runwayml/stable-diffusion-v1-5",
            "mhdang/dpo-sd1.5-text2image-v1",
            "jacklishufan/diffusion-kto",
            "ylwu/diffusion-dro-sd1.5",
            "JaydenLu666/InPO-SD1.5",
        ],
        help="UNet initialization model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a LoRA checkpoint directory (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images and prompt-image mapping JSON.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of generated images (height and width).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    main(args)