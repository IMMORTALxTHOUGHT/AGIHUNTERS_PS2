"""STAGE 4 - Defect Classifier: ViT (fine-tuned on MVTec/NEU/DAGM).

Real algorithm (implement on the box):
  model = vit_b_16(pretrained=True)
  model.heads.head = nn.Linear(768, NUM_CLASSES)
  freeze all but head + last encoder block
  train with CrossEntropyLoss / AdamW for a few epochs
  torch.save(model.state_dict(), "models/vit_defect.pt")

Contract: classify(img_tensor) -> (defect_type:str, class_conf:float)
"""
from __future__ import annotations
import torch


def classify(img_tensor):
    # ---- SCAFFOLD DUMMY (replace with real ViT) ----
    return "scratch", 0.90
