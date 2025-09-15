import torch
import torchvision
from transformers import AutoProcessor, AutoModel
    
class PickScore:
    def __init__(self):
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval()
        
    def requires_grad_(self, require_grad):
        self.model.requires_grad_(require_grad)
        
    def to(self, device, dtype):
        self.model.to(device, dtype=dtype)
        