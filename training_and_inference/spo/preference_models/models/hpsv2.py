import os
import torch
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

class HPSv2:
    def __init__(self):
        model_name = "ViT-H-14"
        self.model, _, _ = create_model_and_transforms(
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
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        