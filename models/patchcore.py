"""STAGE 3 - Vision Detector: PatchCore (anomaly detection + heatmap).

Real algorithm (implement on the box):
  1. Load a pretrained CNN (resnet18 / wide_resnet50_1).
  2. Hook an intermediate layer -> (1, 512, 28, 28) patch-feature map.
  3. build_memory(): run all GOOD images, collect patch vectors, coreset
     subsample, save to mem_bank.npy.  (no backprop - training free)
  4. score(): for each test patch find nearest memory patch via FAISS;
     upsample distances to H x W -> heatmap; anomaly = max(heatmap).

Contract: PatchCore.score(img_tensor) -> (score:float, heatmap:np.ndarray,
                                         roi:(x1,y1,x2,y2), roi_crop:PIL.Image)
"""
from __future__ import annotations
import numpy as np
from PIL import Image


class PatchCore:
    def __init__(self):
        self.mem = None  # np.ndarray (M, 512) loaded from mem_bank.npy

    def build_memory(self, good_dir: str) -> None:
        # TODO: extract patch features from every image in good_dir,
        # coreset-subsample, np.save("models/mem_bank.npy", mem)
        raise NotImplementedError("Run build_memory() on MVTec good/ first.")

    def score(self, img_tensor):
        # ---- SCAFFOLD DUMMY (replace with real PatchCore) ----
        _, _, H, W = img_tensor.shape
        return 0.0, np.zeros((H, W)), (0, 0, 10, 10), Image.new("RGB", (32, 32))
