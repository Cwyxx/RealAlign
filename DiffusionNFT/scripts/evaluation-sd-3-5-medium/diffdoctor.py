from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn as nn
import torch
import cv2
from torchvision import transforms
import numpy as np
import os
from PIL import Image

def get_segformer(path_or_hub, out_channels=1):
    # load a pretrained Segformer model
    preprocessor = SegformerImageProcessor.from_pretrained(path_or_hub)
    model = SegformerForSemanticSegmentation.from_pretrained(path_or_hub)
    # change the number of output channels
    model.decode_head.classifier = nn.Conv2d(model.decode_head.classifier.in_channels, out_channels, kernel_size=1)
    return preprocessor, model

class DiffDoctor(nn.Module):
    def __init__(self, image_dir):
        super().__init__()
        segformer_path = "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/DiffDoctor/ad_pytorch_model.bin"
        method = os.path.basename(os.path.dirname(image_dir))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(image_dir)))
        generated_image_seed = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_dir))))
        print(f"generated_image_seed: {generated_image_seed}")
        print(f"dataset: {dataset}")
        print(f"method: {method}")
        self.output_dir = f"/data_center/data2/dataset/chenwy/21164-data/Artifact_Segmentor_Tmp/visualization/DiffDoctor/{generated_image_seed}/{dataset}/{method}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.device = torch.device("cuda")
        self.seg_preprocessor, self.artifact_detector = get_segformer("nvidia/mit-b5", out_channels=1)
        self.artifact_detector.load_state_dict(torch.load(segformer_path))
        self.artifact_detector.to(self.device)
        self.artifact_detector.eval()
        
    def __call__(self, image_path):
        uid, _ = os.path.splitext((os.path.basename(image_path)))
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        visualizable_image = image.copy()
        with torch.no_grad():
            image = transforms.ToTensor()(image).to(self.device)
            processed_images = self.seg_preprocessor(image, return_tensors='pt',do_rescale=False)['pixel_values'].to(self.device)
            pred = self.artifact_detector(processed_images)
            pred = nn.functional.interpolate(
                pred.logits, size=processed_images.shape[-2:], mode="bilinear", align_corners=False
            )
            normed_preds = torch.sigmoid(pred)
            
        mask = (normed_preds[0].detach().cpu().numpy().transpose(-2, -1, -3) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        alpha = 0.4  # the transparency of the heatmap
        masked_img = cv2.addWeighted(visualizable_image, 1-alpha, heatmap_color, alpha, 0)
        
        output_path = os.path.join(self.output_dir, f"{uid}.png")
        cv2.imwrite(output_path, masked_img)
        
        art_problem_pixels = np.sum(mask >= 128)
        total_pxiel_nums = mask.size
        perceptual_artifact_ratio = art_problem_pixels / total_pxiel_nums

        return perceptual_artifact_ratio