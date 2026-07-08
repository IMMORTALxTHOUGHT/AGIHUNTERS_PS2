import numpy as np
import faiss
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class FaissStore:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.index = None
        self.records: list = []

    def _ensure_index(self):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)

    def add(self, emb: np.ndarray, record: dict) -> None:
        self._ensure_index()
        self.index.add(emb.reshape(1, -1).astype(np.float32))
        self.records.append(record)

    def build(self, embeddings: np.ndarray, records: list) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        assert embeddings.shape[1] == self.dim, (
            f"Expected dim {self.dim}, got {embeddings.shape[1]}"
        )
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.records = list(records)

    def search(self, emb: np.ndarray, k: int = 5) -> list:
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(emb.reshape(1, -1).astype(np.float32), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            rec = dict(self.records[idx])
            rec["similarity"] = float(dist)
            results.append(rec)
        return results

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path + ".index")
        np.save(path + ".records.npy", np.array(self.records, dtype=object), allow_pickle=True)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path + ".index")
        self.records = list(np.load(path + ".records.npy", allow_pickle=True))


def build_store_from_datasets(
    embed_fn,
    mvtec_root="datasets/mvtec/",
    neu_root="datasets/neu/NEU-DET/",
    dagm_root="datasets/dagm/dagm_kaggleupload/DAGM_KaggleUpload/",
    max_per_class: int = 50,
):
    from collections import defaultdict
    import random

    all_pairs = []

    def _collect(root, label_fn):
        root = Path(root)
        for p in root.rglob("*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                try:
                    all_pairs.append((p, label_fn(p)))
                except Exception:
                    continue

    if mvtec_root:
        def mvtec_label(p):
            parts = Path(p).parts
            return f"{parts[-4]}_{parts[-2]}"
        _collect(Path(mvtec_root), mvtec_label)

    if neu_root:
        def neu_label(p):
            return Path(p).parts[-2]
        _collect(Path(neu_root) / "train" / "images", neu_label)
        _collect(Path(neu_root) / "validation" / "images", neu_label)

    if dagm_root:
        def dagm_label(p):
            return Path(p).parts[-3]
        _collect(Path(dagm_root), dagm_label)

    by_label = defaultdict(list)
    for p, lbl in all_pairs:
        by_label[lbl].append(p)

    rng = random.Random(42)
    selected = []
    for lbl, paths in by_label.items():
        for p in rng.sample(paths, min(len(paths), max_per_class)):
            selected.append((p, lbl))

    embeddings = []
    records = []
    for p, lbl in tqdm(selected, desc="Embedding selected"):
        try:
            img = Image.open(p).convert("RGB")
            emb = embed_fn(img)
            embeddings.append(emb)
            records.append({"path": str(p), "label": lbl})
        except Exception:
            continue

    store = FaissStore(dim=embeddings[0].shape[0] if embeddings else 256)
    store.build(np.array(embeddings), records)
    return store
