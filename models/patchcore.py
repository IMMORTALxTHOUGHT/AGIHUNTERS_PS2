import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import faiss


class PatchCore:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = wide_resnet50_2(pretrained=True)
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:7])
        self.backbone.eval().to(self.device)
        self.memory_bank = None
        self.index = None
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract_features(self, img_tensor):
        with torch.no_grad():
            features = self.backbone(img_tensor)
            features = features.permute(0, 2, 3, 1)
            return features.reshape(-1, 1024).cpu().numpy()

    def build_memory(self, mvtec_root: str, coreset_fraction: float = 0.1):
        good_dirs = sorted(Path(mvtec_root).glob("*/train/good"))
        if not good_dirs:
            raise FileNotFoundError(f"No MVTec train/good dirs found under {mvtec_root}")

        all_features = []
        for gd in tqdm(good_dirs, desc="Building memory bank"):
            for img_path in gd.iterdir():
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                    continue
                img = Image.open(img_path).convert("RGB")
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                feats = self._extract_features(tensor)
                all_features.append(feats)

        all_features = np.concatenate(all_features, axis=0).astype(np.float32)
        n = len(all_features)
        n_sample = max(1000, int(n * coreset_fraction))
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=min(n_sample, n), replace=False)
        self.memory_bank = all_features[idx]
        self._build_index()

    def _build_index(self):
        if self.memory_bank is None:
            return
        dim = self.memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.memory_bank)

    def save_memory(self, path: str):
        np.save(path, self.memory_bank)

    def load_memory(self, path: str):
        self.memory_bank = np.load(path).astype(np.float32)
        self._build_index()

    def predict(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        orig_size = image.size
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        feats = self._extract_features(tensor).astype(np.float32)
        patch_h = patch_w = 28

        if self.index is not None and self.index.ntotal > 0:
            D, _ = self.index.search(feats, 1)
            patch_scores = D[:, 0]
        else:
            dists = np.linalg.norm(
                feats[:, None] - self.memory_bank[None, :], axis=2
            )
            patch_scores = dists.min(axis=1)

        anomaly_score = float(patch_scores.max())
        heatmap = patch_scores.reshape(patch_h, patch_w)
        heatmap = cv2.resize(heatmap, orig_size, interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        _, thresh = cv2.threshold(
            heatmap, 0.5, 1.0, cv2.THRESH_BINARY
        )
        thresh = thresh.astype(np.uint8)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            roi = (x, y, x + w, y + h)
            roi_crop = image.crop(roi)
        else:
            roi = (0, 0, 32, 32)
            roi_crop = image.crop(roi)

        heatmap_rgb = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            np.array(image.resize(orig_size)), 0.5, heatmap_colored, 0.5, 0
        )

        return {
            "anomaly_score": anomaly_score,
            "heatmap": heatmap,
            "heatmap_overlay": overlay,
            "roi": roi,
            "roi_crop": roi_crop,
        }
