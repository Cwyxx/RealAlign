import argparse
import os
import json
import logging

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class DPGBenchPromptDataset(Dataset):
    """DPG-Bench: load prompts from dpg_bench/prompts/*.txt files.

    Each .txt file contains a single prompt; the filename stem (e.g.
    'partiprompts97') is used as the item_id and must match the keys in
    dpg_bench.csv so that compute_dpg_bench.py can look up questions.
    """

    def __init__(self, prompts_dir: str):
        self.prompts_dir = os.path.abspath(prompts_dir)
        if not os.path.isdir(self.prompts_dir):
            raise FileNotFoundError(f"DPG-Bench prompts directory not found: {self.prompts_dir}")

        items = []
        for fname in sorted(os.listdir(self.prompts_dir)):
            if not fname.endswith(".txt"):
                continue
            item_id = fname[:-4]  # strip .txt
            fpath = os.path.join(self.prompts_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            if prompt:
                items.append({"item_id": item_id, "prompt": prompt})

        self.items = items
        print(f"Loaded {len(self.items)} prompts from {self.prompts_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(examples):
    item_ids = [e["item_id"] for e in examples]
    prompts = [e["prompt"] for e in examples]
    return item_ids, prompts


def make_grid(images: list, cols: int = 2) -> Image.Image:
    """Tile a list of PIL images into a cols-wide grid."""
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (w * cols, h * rows))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    return grid


def main(args):
    device = torch.device("cuda")

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if args.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None

    print("Running DPG-Bench evaluation with SD-3.5-Medium on 1 GPU.")
    if enable_amp:
        print(f"Using mixed precision: {args.mixed_precision}")

    os.makedirs(args.output_dir, exist_ok=True)

    # compute_dpg_bench.py reads images directly from --image-root-path
    images_dir = args.output_dir
    os.makedirs(images_dir, exist_ok=True)

    # --- Load Model and Pipeline ---
    print("Loading model and pipeline (stabilityai/stable-diffusion-3.5-medium)...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium"
    )

    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(
        r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"args.checkpoint_path: {args.checkpoint_path}")
    if (
        args.checkpoint_path is not None
        and args.checkpoint_path
        and os.path.exists(os.path.join(args.checkpoint_path, "lora", "learner"))
    ):
        lora_path = os.path.join(args.checkpoint_path, "lora", "learner")
        print(f"Loading LoRA weights from: {lora_path}")
        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.load_adapter(
            lora_path, adapter_name="learner", is_trainable=False
        )
        pipeline.transformer.set_adapter("learner")
        print(f"pipeline.transformer.active_adapter: {pipeline.transformer.active_adapter}")

    pipeline.transformer.eval()
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.transformer.to(device, dtype=text_encoder_dtype)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # --- Load DPG-Bench Dataset ---
    prompts_dir = args.prompts_dir
    if not prompts_dir:
        prompts_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dpg_bench", "prompts"
        )
    print(f"Loading DPG-Bench prompts from: {prompts_dir}")
    dataset = DPGBenchPromptDataset(prompts_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
    )

    skipped = 0
    for batch in tqdm(dataloader, desc="Generating images"):
        item_ids, prompts = batch
        item_id = item_ids[0]
        prompt = prompts[0]

        image_path = os.path.join(images_dir, f"{item_id}.png")
        if args.skip_existing and os.path.exists(image_path):
            skipped += 1
            continue

        if args.pic_num == 1:
            # Single image per prompt
            generator = torch.Generator(device).manual_seed(args.seed)
            with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    out = pipeline(
                        [prompt],
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        output_type="pil",
                        height=args.resolution,
                        width=args.resolution,
                        generator=generator,
                    )
            pil_image = out.images[0]
        else:
            # Generate pic_num images and tile them into a 2x2 grid.
            # compute_dpg_bench.py crops 4 quadrants of size resolution×resolution
            # from a 2*resolution × 2*resolution image.
            pil_images = []
            generator = torch.Generator(device).manual_seed(0)
            for _ in range(args.pic_num):
                with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    with torch.no_grad():
                        out = pipeline(
                            [prompt],
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            output_type="pil",
                            height=args.resolution,
                            width=args.resolution,
                            generator=generator,
                        )
                pil_images.append(out.images[0])
                # advance seed so each sub-image differs
                generator = torch.Generator(device).manual_seed(len(pil_images) * args.seed)
            pil_image = make_grid(pil_images, cols=2)

        pil_image.save(image_path)

    print(f"Done. Images saved to: {images_dir}")
    if skipped:
        print(f"Skipped {skipped} already-existing images (--skip_existing).")

    # Save a JSON mapping item_id -> image_path for reference
    mapping = {
        item["item_id"]: os.path.join(images_dir, f"{item['item_id']}.png")
        for item in dataset.items
    }
    mapping_path = os.path.join(args.output_dir, "item_image_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Item-image mapping saved to: {mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SD-3.5-Medium images for DPG-Bench prompts."
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default=None,
        help="Path to dpg_bench/prompts/ directory. Default: <script_dir>/dpg_bench/prompts/.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Local path to the SD3 LoRA checkpoint directory "
            "(expects '<checkpoint_path>/lora/learner')."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Directory to save generated images. "
            "This path is passed directly to compute_dpg_bench.py as --image-root-path."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "Resolution of each generated image. "
            "When --pic_num 4 the saved image is 2×resolution in each dimension."
        ),
    )
    parser.add_argument(
        "--pic_num",
        type=int,
        default=4,
        choices=[1, 4],
        help=(
            "Number of images to generate per prompt. "
            "1 = single image; 4 = 2×2 grid (matches dist_eval.sh default PIC_NUM=4)."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of diffusion inference steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if the output image already exists.",
    )

    args = parser.parse_args()
    main(args)