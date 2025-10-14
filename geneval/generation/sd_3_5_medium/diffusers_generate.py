"""Adapted from TODO"""

import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model, PeftModel

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between 'no', 'fp16', or 'bf16'.",
    )
    return opt


def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # --- Mixed Precision Setup ---
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    mixed_precision_dtype = None
    if opt.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif opt.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None
    
    print(f"Running evaluation with 1 GPUs.")
    if enable_amp: print(f"Using mixed precision: {opt.mixed_precision}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # --- Mixed Precision Setup ---
    
    # --- Load Model and Pipeline ---
    print("Loading model and pipeline (stabilityai/stable-diffusion-3.5-medium)...")
    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", text_encoder_3=None, tokenizer_3=None)
    # pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
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

    if opt.lora_hf_path is not None and opt.lora_hf_path:
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, opt.lora_hf_path)
        pipeline.transformer = pipeline.transformer.merge_and_unload()
        print(f"Loading LoRA weights from HuggingFace: {opt.lora_hf_path}.")
        
    elif opt.checkpoint_path is not None and opt.checkpoint_path and os.path.exists(os.path.join(opt.checkpoint_path, "lora")):
        lora_path = os.path.join(opt.checkpoint_path, "lora")
        print(f"Loading LoRA weights from: {lora_path}")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"LoRA directory not found at {lora_path}. Ensure your checkpoint has a 'lora' subdirectory."
            )
        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.load_adapter(lora_path, adapter_name="default", is_trainable=False)
    pipeline.transformer.eval()
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.transformer.to(device, dtype=text_encoder_dtype)
    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    # pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(position=1, leave=False, desc="Timestep", dynamic_ncols=True,)
    pipeline.to(device)
    # --- Load Model and Pipeline ---
    
    print(f"Output dir: {opt.outdir}")
    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                # Generate images
                samples = pipeline(
                    prompt,
                    height=opt.H,
                    width=opt.W,
                    num_inference_steps=opt.num_inference_steps,
                    guidance_scale=opt.scale,
                    num_images_per_prompt=min(batch_size, opt.n_samples - sample_count),
                    negative_prompt=opt.negative_prompt or None
                )[0]
                
                for sample in samples:
                    image_path = os.path.join(sample_path, f"{sample_count:05}.png")
                    pil_image = Image.fromarray((sample.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil_image.save(image_path)
                    sample_count += 1
                    
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

        del all_samples

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
