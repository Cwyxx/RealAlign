"""U^2-Net saliency mask extraction.

Wraps the vendored U^2-Net so that each inpainting backend can call::

    saliency = U2NetSaliency(ckpt_path, device="cuda")
    mask = saliency(real_image)              # PIL.Image, mode RGB, soft mask

The returned mask is the (resized-back) probability map at the original image
resolution, suitable for ``StableDiffusion*InpaintPipeline.mask_image``.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from u2net_arch import U2NET


def _norm_pred(d: torch.Tensor) -> torch.Tensor:
    ma, mi = torch.max(d), torch.min(d)
    return (d - mi) / (ma - mi)


class U2NetSaliency:
    """Loads U^2-Net once and produces a soft saliency mask per call."""

    def __init__(self, ckpt_path: str, device: str = "cuda", input_size: int = 320):
        self.device = device
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.net = U2NET(3, 1)
        if device == "cuda":
            self.net.load_state_dict(torch.load(ckpt_path))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.net.eval()

    @torch.no_grad()
    def __call__(self, real_image: Image.Image) -> Image.Image:
        original_size = real_image.size  # (W, H)
        x = self.transform(real_image).unsqueeze(0).to(self.device)
        d1, *_ = self.net(x)
        pred = _norm_pred(d1[:, 0, :, :])
        pred_np = pred.squeeze().cpu().numpy()
        pred_pil = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_pil = pred_pil.resize(original_size, resample=Image.BILINEAR).convert("RGB")
        return pred_pil


def binarize_mask(mask: Image.Image, threshold: int = 128) -> Image.Image:
    """Convert a soft RGB saliency map to a binary L-mode mask (for archival).

    The pipelines themselves are fed the soft mask; the binarized version is
    only saved to disk for inspection / qualitative analysis.
    """
    arr = np.array(mask.convert("L"))
    return Image.fromarray(((arr > threshold).astype(np.uint8) * 255), mode="L")
