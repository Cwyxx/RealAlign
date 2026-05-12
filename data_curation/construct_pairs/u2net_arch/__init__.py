"""Vendored U^2-Net model definition.

Source: https://github.com/xuebinqin/U-2-Net (Apache-2.0)
Reference: Qin et al., "U^2-Net: Going Deeper with Nested U-Structure for
Salient Object Detection", Pattern Recognition 2020.

Only ``u2net.py`` is vendored here; the upstream ``u2net_refactor.py`` and
training code are not needed for inference-time saliency extraction.
"""

from .u2net import U2NET, U2NETP

__all__ = ["U2NET", "U2NETP"]
