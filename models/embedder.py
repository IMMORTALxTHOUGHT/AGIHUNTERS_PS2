import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


class Embedder:
    def __init__(self, vit_model):
        self.vit = vit_model
        self.vit.eval()
        self.device = next(vit_model.parameters()).device
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.vit.backbone(tensor)
            embedding = F.normalize(self.vit.proj(features), dim=1)
        return embedding[0].cpu().numpy().astype(np.float32)

    def encode_tensor(self, tensor):
        with torch.no_grad():
            features = self.vit.backbone(tensor)
            embedding = F.normalize(self.vit.proj(features), dim=1)
        return embedding.cpu().numpy().astype(np.float32)
