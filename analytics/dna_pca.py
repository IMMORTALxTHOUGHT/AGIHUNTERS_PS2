"""STAGE 9 - Defect DNA (PCA 2D embedding space).

Contract: project(embeddings: np.ndarray (N,768)) -> np.ndarray (N,2)
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


def project(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], 2))
    return PCA(2).fit_transform(embeddings)
