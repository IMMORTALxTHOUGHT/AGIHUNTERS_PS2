import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import timm
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T


CLASS_MAPPING = {}
_idx = 0

def _register(name):
    global _idx
    if name not in CLASS_MAPPING:
        CLASS_MAPPING[name] = _idx
        _idx += 1
    return CLASS_MAPPING[name]


def _collect_mvtec(root):
    samples = []
    for cat_dir in sorted(Path(root).iterdir()):
        if not cat_dir.is_dir():
            continue
        train_good = cat_dir / "train" / "good"
        if train_good.exists():
            for p in train_good.iterdir():
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    samples.append((str(p), _register(f"{cat_dir.name}_good")))
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for defect_dir in test_dir.iterdir():
                if not defect_dir.is_dir():
                    continue
                cls_name = f"{cat_dir.name}_{defect_dir.name}"
                for p in defect_dir.iterdir():
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                        samples.append((str(p), _register(cls_name)))
    return samples


def _collect_neu(root):
    samples = []
    images_dir = Path(root) / "train" / "images"
    if not images_dir.exists():
        return samples
    for cls_dir in sorted(images_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = _register(cls_dir.name)
        for p in cls_dir.iterdir():
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                samples.append((str(p), label))
    val_images_dir = Path(root) / "validation" / "images"
    if val_images_dir.exists():
        for cls_dir in sorted(val_images_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = _register(cls_dir.name)
            for p in cls_dir.iterdir():
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    samples.append((str(p), label))
    return samples


def _collect_dagm(root):
    samples = []
    base = Path(root)
    for cls_dir in sorted(base.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = _register(cls_dir.name)
        train_dir = cls_dir / "Train"
        if train_dir.exists():
            for p in train_dir.iterdir():
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    samples.append((str(p), label))
        test_dir = cls_dir / "Test"
        if test_dir.exists():
            for p in test_dir.iterdir():
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    samples.append((str(p), label))
    return samples


class DefectDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=None, model_name="vit_base_patch16_224"):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.num_features
        self.proj = nn.Linear(embed_dim, 256)
        self.classifier = nn.Linear(embed_dim, num_classes or 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        embeds = F.normalize(self.proj(features), dim=1)
        return logits, embeds

    def predict(self, image, label_map=None):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0)
        if next(self.parameters()).is_cuda:
            tensor = tensor.cuda()
        self.eval()
        with torch.no_grad():
            logits, embeds = self.forward(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            label_idx = int(idx.item())
            confidence = float(conf.item())
            if label_map:
                label_str = label_map[label_idx]
            else:
                label_str = str(label_idx)
            return {
                "label": label_str,
                "confidence": confidence,
                "embedding": embeds[0].cpu().numpy(),
                "probs": probs[0].cpu().numpy(),
            }


def load_datasets(mvtec_root, neu_root, dagm_root):
    samples = []
    if mvtec_root:
        samples.extend(_collect_mvtec(mvtec_root))
    if neu_root:
        samples.extend(_collect_neu(neu_root))
    if dagm_root:
        samples.extend(_collect_dagm(dagm_root))
    return samples, {v: k for k, v in CLASS_MAPPING.items()}


def train(
    mvtec_root="datasets/mvtec/",
    neu_root="datasets/neu/NEU-DET/",
    dagm_root="datasets/dagm/dagm_kaggleupload/DAGM_KaggleUpload/",
    output_path="models/weights/vit_defect.pt",
    epochs=30,
    batch_size=32,
    lr=1e-4,
    val_split=0.2,
):
    global _idx, CLASS_MAPPING
    _idx = 0
    CLASS_MAPPING = {}

    samples, label_map = load_datasets(mvtec_root, neu_root, dagm_root)
    num_classes = len(label_map)
    print(f"Loaded {len(samples)} samples across {num_classes} classes")
    print(f"Classes: {list(label_map.values())}")

    dataset = DefectDataset(samples)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTClassifier(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "label_map": label_map,
                "val_acc": val_acc,
                "epoch": epoch,
            }, output_path)
            print(f"Saved best model (acc={val_acc:.4f}) to {output_path}")

    print(f"Training complete. Best val_acc={best_acc:.4f}")
    return output_path


def load_model(path="models/weights/vit_defect.pt", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    label_map = ckpt["label_map"]
    model = ViTClassifier(num_classes=len(label_map))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, label_map
