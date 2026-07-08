"""Inference — single-image run tying PatchCore + ViT + Embedder + FAISS.

Dashboard and CLI both call infer_one().
"""
import numpy as np
from PIL import Image

from config import (
    VIT_CONF_THRESHOLD,
    FAISS_TOP_K,
    MODEL_WEIGHTS,
)
from models.patchcore import PatchCore
from models.vit_classifier import load_model
from models.embedder import Embedder
from storage.faiss_store import FaissStore, build_store_from_datasets


_pc = None
_model = None
_emb = None
_label_map = None
_store = None


def _init_models():
    global _pc, _model, _emb, _label_map, _store
    if _pc is None:
        _pc = PatchCore()
        _pc.load_memory(str(MODEL_WEIGHTS / "mem_bank.npy"))
        _model, _label_map = load_model(str(MODEL_WEIGHTS / "vit_defect.pt"))
        _emb = Embedder(_model)

        store_path = MODEL_WEIGHTS / "faiss_store"
        if store_path.with_suffix(".index").exists():
            _store = FaissStore()
            _store.load(str(store_path))
        else:
            print("Building FAISS store from datasets (one-time)...")
            _store = build_store_from_datasets(_emb.encode)
            _store.save(str(store_path))


def infer_one(image_path: str) -> dict:
    _init_models()

    img = Image.open(image_path).convert("RGB")
    result = _pc.predict(img)
    cls = _model.predict(img, _label_map)
    vec = _emb.encode(img)

    is_novel = cls["confidence"] < VIT_CONF_THRESHOLD
    similar = _store.search(vec, k=FAISS_TOP_K) if _store else []

    return {
        "anomaly_score": result["anomaly_score"],
        "heatmap": result["heatmap"],
        "heatmap_overlay": result["heatmap_overlay"],
        "roi": result["roi"],
        "defect": cls["label"],
        "vit_confidence": cls["confidence"],
        "is_novel_defect": is_novel,
        "embedding": vec,
        "similar_cases": similar,
    }


def reset():
    global _pc, _model, _emb, _label_map, _store
    _pc = _model = _emb = _label_map = _store = None
