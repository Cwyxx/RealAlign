import os
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from torchvision import transforms
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from model import U2NET

"""Normalize the predicted SOD probability map"""
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def get_saliency_mask(real_image, net, input_size=320, device="cuda"):
    # 1. PIL.Image -> convert to RGB
    original_shape = real_image.size  # (width, height)
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    inputs_test = transform_pipeline(real_image).unsqueeze(0).to(device)
    
    
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        pred_np = pred.squeeze().cpu().data.numpy()
        
        pred_pil = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_pil = pred_pil.resize(original_shape, resample=Image.BILINEAR)
        pred_pil = pred_pil.convert('RGB')
    
    return pred_pil

def parse_args():
    parser = argparse.ArgumentParser(description="Generate real-fake image pairs")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing real images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save real-fake pairs",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="CSV file stores the uid and corresponding prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for StableDiffusionInpaintPipeline.",
    )
    
    return parser.parse_args()

def main(args):
    # Check if prompt_file is provided
    if args.prompt_file is None:
        raise ValueError("--prompt_file is required")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    u2net_model_path = "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/u2net/u2net.pth"
    print("...load U2NET---173.6 MB")
    net = U2NET(3, 1)
    
    # Load model with appropriate map_location
    if device == "cuda":
        net.load_state_dict(torch.load(u2net_model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(u2net_model_path, map_location='cpu'))
    net.eval()

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
    
    df = pd.read_csv(args.prompt_file, dtype=str)
    total_uids = len(df)
    print(f"Total UIDs in prompt file: {total_uids}")
    
    processed_count = 0
    skipped_count = 0
    resize_512_center_crop = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(512),
    ])
    
    for idx, row in tqdm(df.iterrows(), total=total_uids, desc="Processing images"):
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
            saliency_mask = get_saliency_mask(real_image, net, device=device)
            fake_image = pipeline(prompt=prompt, image=real_image, mask_image=saliency_mask).images[0]
            
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