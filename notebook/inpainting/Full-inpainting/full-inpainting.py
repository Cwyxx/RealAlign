import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from torchvision import transforms
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

"""Create a full mask (all white): entire image is the inpainting region."""
def get_full_mask(image):
    w, h = image.size
    mask_np = np.ones((h, w), dtype=np.uint8) * 255
    return Image.fromarray(mask_np, mode='L')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate real-fake image pairs")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/civitai-top-sfw-images-with-metadata/images",
        help="Directory containing real images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/civitai-top-sfw-images-with-metadata",
        help="Directory to save real-fake pairs",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/civitai-top-sfw-images-with-metadata/uid_prompt.csv",
        help="CSV file stores the uid and corresponding prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for StableDiffusionInpaintPipeline.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index for processing (0-based). If not specified, starts from the beginning.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for processing (exclusive). If not specified, processes until the end.",
    )
    
    return parser.parse_args()

def main(args):
    # Check if prompt_file is provided
    if args.prompt_file is None:
        raise ValueError("--prompt_file is required")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Full inpainting: entire image as mask (no U2Net saliency).")

    pipeline = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting")
    pipeline = pipeline.to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(leave=False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    real_image_output_dir = os.path.join(args.output_dir, "real")
    fake_image_output_dir = os.path.join(args.output_dir, "fake")
    os.makedirs(real_image_output_dir, exist_ok=True)
    os.makedirs(fake_image_output_dir, exist_ok=True)

    print(f"Processing images from {args.input_dir} to {args.output_dir}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"Real image output directory: {real_image_output_dir}")
    print(f"Fake image output directory: {fake_image_output_dir}")
    
    processed_uids = set()
    if os.path.exists(fake_image_output_dir):
        for f in os.listdir(fake_image_output_dir):
            if f.endswith('.png'):
                uid = os.path.splitext(f)[0]
                processed_uids.add(uid)
    
    print(f"Found {len(processed_uids)} already processed UIDs")
    
    df = pd.read_csv(args.prompt_file, dtype={'uid': str})
    total_uids = len(df)
    
    # Apply start_index and end_index if provided
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else total_uids
        df = df.iloc[start_idx:end_idx]
        print(f"Processing range: [{start_idx}:{end_idx}] (total: {total_uids})")
        print(f"Actual processing: {len(df)} items")
    else:
        print(f"Total UIDs in prompt file: {total_uids}")
    
    processed_count = 0
    skipped_count = 0
    resize_512_center_crop = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(512),
    ])
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        uid = row['uid']
        prompt = row['prompt']
        
        if uid in processed_uids:
            skipped_count += 1
            continue
        
        image_path = None
        image_extensions = ['.jpg', '.jpeg', '.png', '.PNG', '.JPEG', '.JPG']
        for ext in image_extensions:
            potential_path = os.path.join(args.input_dir, f"{uid}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"\nWarning: Image not found for uid {uid}, skipping...")
            continue
        
        try:
            real_image = resize_512_center_crop(Image.open(image_path).convert("RGB"))
            full_mask = get_full_mask(real_image)
            fake_image = pipeline(prompt=prompt, image=real_image, mask_image=full_mask).images[0]
            
            real_image_output_path = os.path.join(real_image_output_dir, f"{uid}.png")
            fake_image_output_path = os.path.join(fake_image_output_dir, f"{uid}.png")
            real_image.save(real_image_output_path)
            fake_image.save(fake_image_output_path)
        
        except Exception as e:
            print(f"\nError processing uid {uid}: {str(e)}")
            continue
        
        processed_count += 1
        
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Total in dataset: {total_uids}")

if __name__ == "__main__":
    args = parse_args()
    main(args)