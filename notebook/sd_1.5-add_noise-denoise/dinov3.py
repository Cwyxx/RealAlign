import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def load_dinov3_model(model_name="facebook/dinov2-base", device=None):
    """
    Args:
            - "facebook/dinov2-base" (DINOv2)
            - "facebook/dinov3-vitb16-pretrain-lvd1689m" (DINOv3 ViT-B/16)
            - "facebook/dinov3-vits14-pretrain-lvd1689m" (DINOv3 ViT-S/14)
            - "facebook/dinov3-vitl14-pretrain-lvd1689m" (DINOv3 ViT-L/14)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Successfully loaded model: {model_name} to device: {device}")
    return model, processor, device

def extract_patch_tokens(model, processor, image, device):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
    
    patch_features_flat = last_hidden_states[:, 1:, :].squeeze(0)
    # print(f"patch_features_flat shape: {patch_features_flat.shape}") # [num_patches, hidden_dim]
    return patch_features_flat


def compute_patch_tokens_difference(tokens1, tokens2):
    assert tokens1.shape == tokens2.shape, f"Tokens shape mismatch: {tokens1.shape} vs {tokens2.shape}"
    
    # calculate the L2 difference between two patch tokens
    differences = torch.norm(tokens1 - tokens2, dim=-1)  # [num_patches]
    max_difference = differences.max().item()
        
    
    return max_difference, differences.cpu().numpy()


def compare_images(model, processor, image1_path, image2_path, device=None):
    tokens1 = extract_patch_tokens(model, processor, image1_path, device)
    tokens2 = extract_patch_tokens(model, processor, image2_path, device)
    
    max_diff, per_patch_diff = compute_patch_tokens_difference(tokens1, tokens2)
    
    return max_diff


if __name__ == "__main__":        
    csv_file_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset/high_quality_train.csv"
    df = pd.read_csv(csv_file_path)
    real_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step/real"
    fake_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step/fake"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, device = load_dinov3_model(model_name="facebook/dinov2-base", device=device)
    
    max_diff_list = []
    for i in tqdm(range(0, len(df))):
        uid = df.iloc[i]["uid"]
        real_image_path = os.path.join(real_image_dir, f"{uid}.png")
        fake_image_path = os.path.join(fake_image_dir, f"{uid}.png")
        max_diff = compare_images(model, processor, real_image_path, fake_image_path, device)
        max_diff_list.append({"uid": uid, "max_diff": max_diff})
    pd.DataFrame(max_diff_list).to_csv("dinov2_max_diff.csv", index=False)