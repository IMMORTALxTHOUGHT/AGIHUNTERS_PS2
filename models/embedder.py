"""STAGE 5 - Embedder (ViT CLS token) for FAISS + Defect DNA.

Real algorithm: load the SAME vit_b_16, strip the head (nn.Identity),
take the CLS token output (768-d), L2-normalize. No extra training.

Contract: embed(img_tensor) -> np.ndarray (768,) float32, L2-normalized
"""
from __future__ import annotations
import numpy as np
import torch


def embed(img_tensor) -> np.ndarray:
    # ---- SCAFFOLD DUMMY (replace with real ViT CLS embedding) ----
    e = np.zeros(768, dtype="float32")
    return e / (np.linalg.norm(e) + 1e-9)
