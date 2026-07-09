"""Inference — single-image run tying PatchCore + ViT + Embedder + FAISS.

Dashboard and CLI both call infer_one().
"""
from config import (
    OOD_CONF_THRESHOLD,
    NOVELTY_CONF_THRESHOLD,
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

    # search one extra so we can drop the queried image itself (it always
    # scores 1.000 as its own nearest neighbour) without losing 5 real cases
    raw = _store.search(vec, k=FAISS_TOP_K + 1) if _store else []
    similar = [s for s in raw if s.get("path") != image_path][:FAISS_TOP_K]

    # ---- closed-set verdict with out-of-distribution rejection ----
    # The ViT softmax max is a RELATIVE score: it always picks the least-wrong
    # known class even for an unrelated image (a monkey, a random Google screw).
    # We therefore grade it in three bands instead of trusting the label.
    conf = float(cls["confidence"])
    if conf < OOD_CONF_THRESHOLD:
        # Not close to anything we were trained on -> do NOT fabricate a class.
        defect = "unknown_part"
        is_ood = True
        is_novel = True
    elif conf < NOVELTY_CONF_THRESHOLD:
        # Real-looking part but an unrecognized variant -> novelty, needs review.
        defect = cls["label"]
        is_ood = False
        is_novel = True
    else:
        defect = cls["label"]
        is_ood = False
        is_novel = False

    return {
        "anomaly_score": result["anomaly_score"],
        "heatmap": result["heatmap"],
        "heatmap_overlay": result["heatmap_overlay"],
        "roi": result["roi"],
        "defect": defect,
        "vit_confidence": conf,
        "is_novel_defect": is_novel,
        "is_ood": is_ood,
        "vit_label": cls["label"],
        "embedding": vec,
        "similar_cases": similar,
        "metadata": meta,
    }


def reset():
    global _pc, _model, _emb, _label_map, _store
    _pc = _model = _emb = _label_map = _store = None
