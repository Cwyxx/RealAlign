import argparse
import logging
import os

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


SUPPORTED_DATASETS = {
    "partiprompts",
    "drawbench-unique",
    "pick_a_pic_v2",
}


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


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    indices = [example["original_index"] for example in examples]
    return prompts, metadatas, indices


def build_dataset(dataset_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(script_dir, "../../dataset", dataset_name))
    print(f"Loading dataset from: {dataset_path}")

    return TextPromptDataset(dataset_path, split="test")


def load_pipeline(args, device):
    print(f"Loading SD-1.5 pipeline with UNet init: {args.unet_init}")
    unet = UNet2DConditionModel.from_pretrained(args.unet_init, subfolder="unet")
    pipeline_kwargs = {"unet": unet}
    if args.mixed_precision == "fp16":
        pipeline_kwargs["torch_dtype"] = torch.float16
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        **pipeline_kwargs,
    )

    if args.checkpoint_path:
        lora_file = os.path.join(args.checkpoint_path, "pytorch_lora_weights.safetensors")
        if not os.path.exists(lora_file):
            raise FileNotFoundError(
                f"LoRA weights not found at {lora_file}. "
                "SD-1.5 checkpoints should contain pytorch_lora_weights.safetensors."
            )
        print(f"Loading LoRA weights from: {args.checkpoint_path}")
        pipeline.load_lora_weights(args.checkpoint_path, weight_name="pytorch_lora_weights.safetensors")
        print(f"Active LoRA adapters: {pipeline.get_active_adapters()}")

    pipeline.to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    return pipeline


def main(args):
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    if args.save_images:
        os.makedirs(images_dir, exist_ok=True)

    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")

    pipeline = load_pipeline(args, device)
    dataset = build_dataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    results = []
    for batch in tqdm(dataloader, desc="Generating images"):
        prompts, metadata, indices = batch
        generator = torch.Generator(device).manual_seed(args.seed)

        with torch.no_grad():
            images = pipeline(
                prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                output_type="pt",
                height=args.resolution,
                width=args.resolution,
                generator=generator,
            )[0]

        for i, sample_idx in enumerate(indices):
            result_item = {
                "sample_id": sample_idx,
                "prompt": prompts[i],
                "metadata": metadata[i] if metadata else {},
                "scores": {},
            }

            if args.save_images:
                image_path = os.path.join(images_dir, f"{sample_idx:05d}.png")
                pil_image = Image.fromarray(
                    (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil_image.save(image_path)
                result_item["image_path"] = image_path

            results.append(result_item)

    results.sort(key=lambda x: x["sample_id"])
    with open(results_filepath, "w") as f_out:
        for result_item in results:
            f_out.write(json.dumps(result_item) + "\n")

    with open(results_filepath.replace(".jsonl", "_backup.jsonl"), "w") as f_out:
        for result_item in results:
            f_out.write(json.dumps(result_item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SD-1.5 images for RealAlign evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="LoRA checkpoint directory containing pytorch_lora_weights.safetensors.",
    )
    parser.add_argument(
        "--unet_init",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="UNet initialization model or local path.",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-v1-5/generate_images",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16"],
    )
    args = parser.parse_args()
    main(args)
