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
        "--input_image_path",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        required=True,
        help="Path to save the output image",
    )
    return parser.parse_args()

def main(args):
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

    resize_512_center_crop = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(512),
    ])
    
    real_image = resize_512_center_crop(Image.open(args.input_image_path).convert("RGB"))
    saliency_mask = get_saliency_mask(real_image, net, device=device)
    saliency_mask.save(args.output_image_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    