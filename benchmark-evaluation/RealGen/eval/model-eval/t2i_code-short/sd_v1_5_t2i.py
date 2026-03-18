import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm


class PromptDataset(Dataset):
    def __init__(self, file_path):
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            self.prompts = [
                'portrait of a woman',
                'A realistic portrait of a young woman, clear facial features, smooth black hair, natural makeup, wearing casual clothes, photographed in soft natural light.',
            ]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def main(args):
    device = torch.device("cuda")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    # Load model and pipeline
    print(f"Loading model from: {args.unet_init}")
    if args.unet_init != "runwayml/stable-diffusion-v1-5":
        unet = UNet2DConditionModel.from_pretrained(args.unet_init, subfolder="unet")
        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(args.unet_init)

    # Load LoRA weights if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
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

    # Load dataset
    print(f"Loading prompts from: {args.prompt_file}")
    dataset = PromptDataset(args.prompt_file)
    print(f"Total prompts: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # Generate images
    for idx, prompt in enumerate(tqdm(dataloader, desc="Generating images")):
        prompt = prompt[0]
        generator = torch.Generator(device).manual_seed(args.seed + idx * 1000)

        with torch.no_grad():
            image = pipeline(
                prompt,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]

        image_path = os.path.join(args.output_dir, "images", f"{idx:05d}.png")
        image.save(image_path)

    print(f"Generation completed! Images saved to: {args.output_dir}/images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using SD-v1.5 for RealBench evaluation")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to text file containing prompts (one per line)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of generated images"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance"
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
            "JaydenLu666/InPO-SD1.5"
        ],
        help="UNet initialization model"
    )
    args = parser.parse_args()
    main(args)
