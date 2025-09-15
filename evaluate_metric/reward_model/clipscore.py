import os
import torch
import torchvision
import torch.nn as nn
import clip

def calculate_clipscore_loss(batch, image, reward_model, device):
    _transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    caption = clip.tokenize(batch["input_text"], truncate=True).to(device)
    txt_features = nn.functional.normalize(reward_model.clip_model.encode_text(caption)) # [Batch_size, 768]
    image = _transform(image).to(device)  # [Batch_size, 3, 224, 224]
    image_features = nn.functional.normalize(reward_model.clip_model.encode_image(image)) # [Batch_size, 768]
    
    reward = torch.sum(torch.mul(txt_features, image_features), dim=1, keepdim=True) # [Batch_size, 1]
    loss = 1.0 - reward
    return reward, loss

class CLIPScore:
    def __init__(self):
        self.clip_model, self.preprocess = clip.load("ViT-L/14", jit=False)
        
    def requires_grad_(self, require_grad):
        self.clip_model.requires_grad_(require_grad)
        self.clip_model.logit_scale.requires_grad_(require_grad)
        
    def to(self, device, dtype):
        self.clip_model.to(device, dtype=dtype)