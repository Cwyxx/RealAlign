import os
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from torchvision import transforms
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import random

def get_sam_mask(real_image, sam_generator, top_k=5):
    """
    使用 SAM 生成 masks，并从面积最大的前 top_k 个 mask 中随机选一个
    
    Args:
        real_image: PIL Image 对象 (RGB)
        sam_generator: SAM mask generation pipeline
        top_k: 选择面积最大的前 k 个 mask
    
    Returns:
        PIL Image: 选中的 mask (RGB 格式)
    """
    # 使用 SAM 生成 masks
    outputs = sam_generator(real_image)
    masks = outputs["masks"]
    
    if len(masks) == 0:
        # 如果没有生成 mask，返回一个全黑的 mask
        return Image.new('RGB', real_image.size, (0, 0, 0))
    
    # 计算每个 mask 的面积
    mask_areas = []
    for mask in masks:
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = mask
        
        # 转换为二值 mask
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0] if mask_array.shape[2] == 1 else mask_array[:, :, 0]
        
        if mask_array.max() > 1:
            mask_array = (mask_array > 127).astype(np.uint8)
        else:
            mask_array = (mask_array > 0.5).astype(np.uint8)
        
        # 计算面积（非零像素的数量）
        area = np.sum(mask_array > 0)
        mask_areas.append((area, mask))
    
    # 按面积排序，选择前 top_k 个
    mask_areas.sort(key=lambda x: x[0], reverse=True)
    top_masks = mask_areas[:min(top_k, len(mask_areas))]
    
    # 从前 top_k 个中随机选一个
    selected_area, selected_mask = random.choice(top_masks)
    
    # 转换为 PIL Image 并确保是 RGB 格式
    if isinstance(selected_mask, Image.Image):
        mask_pil = selected_mask
    else:
        mask_array = selected_mask
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0] if mask_array.shape[2] == 1 else mask_array[:, :, 0]
        
        if mask_array.max() > 1:
            mask_array = (mask_array > 127).astype(np.uint8) * 255
        else:
            mask_array = (mask_array > 0.5).astype(np.uint8) * 255
        
        mask_pil = Image.fromarray(mask_array)
    
    # 确保 mask 和原图尺寸一致
    if mask_pil.size != real_image.size:
        mask_pil = mask_pil.resize(real_image.size, resample=Image.BILINEAR)
    
    # 转换为 RGB 格式
    mask_pil = mask_pil.convert('RGB')
    
    return mask_pil

def parse_args():
    parser = argparse.ArgumentParser(description="Generate real-fake image pairs using SAM")
    parser.add_argument(
        "--input_dir",
        type=str,
        required="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/real",
        help="Directory containing real images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/sam_next_inpainting/random_top_5_mask",
        help="Directory to save real-fake pairs",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/no_anime_all_images.csv",
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
        help="Start index for processing (0-based). If None, start from beginning.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for processing (exclusive). If None, process to the end.",
    )
    
    return parser.parse_args()

def main(args):
    # Check if prompt_file is provided
    if args.prompt_file is None:
        raise ValueError("--prompt_file is required")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1
    print(f"Using device: {device}")
    
    # Load SAM model
    sam_model = "facebook/sam-vit-huge"
    print(f"...load SAM model: {sam_model}")
    sam_generator = pipeline(
        "mask-generation", 
        model=sam_model, 
        device=device_id, 
        framework="pt"
    )
    
    # Load inpainting pipeline
    print("...load StableDiffusionInpaintPipeline")
    inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting"
    )
    inpaint_pipeline = inpaint_pipeline.to(device)
    inpaint_pipeline.safety_checker = None
    inpaint_pipeline.set_progress_bar_config(leave=False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    real_image_output_dir = os.path.join(args.output_dir, "real")
    fake_image_output_dir = os.path.join(args.output_dir, "fake")
    mask_output_dir = os.path.join(args.output_dir, "mask")
    os.makedirs(real_image_output_dir, exist_ok=True)
    os.makedirs(fake_image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    print(f"Processing images from {args.input_dir} to {args.output_dir}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"Real image output directory: {real_image_output_dir}")
    print(f"Fake image output directory: {fake_image_output_dir}")
    print(f"Mask output directory: {mask_output_dir}")
    
    # Handle start_index and end_index
    df = pd.read_csv(args.prompt_file, dtype=str)
    total_uids = len(df)
    
    start_idx = args.start_index if args.start_index is not None else 0
    end_idx = args.end_index if args.end_index is not None else total_uids
    
    # Validate indices
    start_idx = max(0, min(start_idx, total_uids))
    end_idx = max(start_idx, min(end_idx, total_uids))
    
    # Slice dataframe if needed
    if start_idx > 0 or end_idx < total_uids:
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"Processing range: [{start_idx}, {end_idx}) (total: {end_idx - start_idx} items)")
    else:
        print(f"Processing all items: {total_uids}")
    
    print(f"Total UIDs to process: {len(df)}")
    
    processed_uids = set()
    if os.path.exists(fake_image_output_dir):
        for f in os.listdir(fake_image_output_dir):
            if f.endswith('.png'):
                uid = os.path.splitext(f)[0]
                processed_uids.add(uid)
    
    print(f"Found {len(processed_uids)} already processed UIDs")
    
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
            saliency_mask = get_sam_mask(real_image, sam_generator, top_k=5)
            fake_image = inpaint_pipeline(
                prompt=prompt, 
                image=real_image, 
                mask_image=saliency_mask,
                num_inference_steps=args.num_inference_steps
            ).images[0]
            
            real_image_output_path = os.path.join(real_image_output_dir, f"{uid}.png")
            fake_image_output_path = os.path.join(fake_image_output_dir, f"{uid}.png")
            mask_output_path = os.path.join(mask_output_dir, f"{uid}.png")
            real_image.save(real_image_output_path)
            fake_image.save(fake_image_output_path)
            saliency_mask.save(mask_output_path)
        
        except Exception as e:
            print(f"\nError processing uid {uid}: {str(e)}")
            continue
        
        processed_count += 1
        
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Total in processing range: {len(df)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

