"""Inference — single-image run tying PatchCore + ViT + Embedder + FAISS.

Dashboard and CLI both call infer_one().
"""
from config import (
    VIT_CONF_THRESHOLD,
    FAISS_TOP_K,
    MODEL_WEIGHTS,
)
from data.loaders import load_image
from data.metadata import make_metadata
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

    img_pil, _, _ = load_image(image_path)
    meta = make_metadata(image_path)

    result = _pc.predict(img_pil)
    cls = _model.predict(img_pil, _label_map)
    vec = _emb.encode(img_pil)

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
        "metadata": meta,
    }


def reset():
    global _pc, _model, _emb, _label_map, _store
    _pc = _model = _emb = _label_map = _store = None
