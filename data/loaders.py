"""STAGE 1 - Image Upload.

Loads an image and produces the two preprocess variants the pipeline needs:
  * img_np  : (H, W, 3) uint8  - for OpenCV / display / heatmap overlay
  * x_vit   : (1, 3, 224, 224)  - normalized tensor for ViT / PatchCore

Contract: load_image(path) -> (img_pil, img_np, x_vit)
"""
from __future__ import annotations
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

VIT_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_image(path: str):
    img_pil = Image.open(path).convert("RGB")
    img_np = np.array(img_pil)                  # (H, W, 3) uint8
    x_vit = VIT_TF(img_pil).unsqueeze(0)        # (1, 3, 224, 224)
    return img_pil, img_np, x_vit


# TODO (data owner): add MVTec / NEU / DAGM training iterators here, e.g.
#   def iter_good(root): ...      # yields normal images for PatchCore memory
#   def iter_labeled(root): ...    # yields (image, label) for ViT fine-tune
