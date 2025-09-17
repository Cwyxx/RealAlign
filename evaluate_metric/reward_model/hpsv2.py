import os
import torch
import torchvision
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

class HPSv2:
    def __init__(self):
        model_name = "ViT-H-14"
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        ) 
        self.tokenizer = get_tokenizer(model_name)
        self.target_size = 224
        checkpoint_path="/data_center/data2/dataset/chenwy/21164-data/model-ckpt/hpsv2/HPS_v2_compressed.pt"
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
    def requires_grad_(self, require_grad):
        self.model.requires_grad_(require_grad)
        
    def to(self, device, dtype):
        self.model.to(device, dtype=dtype)