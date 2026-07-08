"""Inference helpers — single-image and batch run.

Pulls together all pipeline stages into a callable that
dashboard and CLI can both use.
"""
from __future__ import annotations

from config import VIT_CONF_THRESHOLD
from data.loaders import load_image
from data.metadata import make_metadata
from models.patchcore import PatchCore
from models.vit_classifier import classify
from models.fewshot import FewShotClassifier
from models.embedder import embed
from storage.faiss_store import FaissStore
from storage.knowledge_graph import KnowledgeGraph
from storage.memory import Memory
from agents.debate import run_debate
from agents.moderator import moderate
from analytics.dna_pca import project
from analytics.calibration import calibrate
from analytics.health import health

# Module-level singletons (initialised once)
_store = FaissStore()
_kg = KnowledgeGraph()
_fewshot = FewShotClassifier()
_memory = Memory()


def infer_one(image_path: str, human_feedback: dict | None = None) -> dict:
    """Run full pipeline on a single image. Returns analysis dict."""
    # 1-2
    img_pil, img_np, x_vit = load_image(image_path)
    meta = make_metadata(image_path)
    # 3
    pc = PatchCore()
    score, heat, roi, crop = pc.score(x_vit)
    # 4
    defect, conf = classify(x_vit)
    # 4b
    is_novel = False
    fewshot_distances = {}
    if conf < VIT_CONF_THRESHOLD:
        emb_fs = embed(x_vit)
        defect, fs_conf, is_novel, fewshot_distances = _fewshot.classify(
            emb_fs, conf, VIT_CONF_THRESHOLD
        )
        conf = fs_conf if fs_conf > conf else conf
    # 5
    emb = embed(x_vit)
    # 6-7
    cases = _store.search(emb, k=5)
    chains = _kg.chains_from(defect)
    # 8
    votes = run_debate({"defect": defect, "heat": heat.shape, "is_novel": is_novel},
                       cases, meta, chains)
    rca = moderate(votes)
    # 9-11
    dna = project(emb.reshape(1, -1))
    cal = calibrate(conf, _fake_distances(), votes, meta)
    h = health([{"meta": meta, "is_defect": score > 0.5}])
    # 12
    record = {"defect_type": defect, "metadata": meta, "rca": rca,
              "anomaly_score": score, "is_novel": is_novel}
    _memory.update(emb, record, fewshot=_fewshot, human_feedback=human_feedback)
    return {
        "anomaly_score": score,
        "defect": defect,
        "vit_confidence": conf,
        "fewshot_fallback": conf < VIT_CONF_THRESHOLD,
        "is_novel_defect": is_novel,
        "fewshot_distances": fewshot_distances,
        "metadata": meta, "rca": rca,
        "dna": dna.tolist(),
        "calibration": cal, "health": h,
    }


def _fake_distances():
    import numpy as np
    return np.array([0.1, 0.2])


def reset():
    """Re-initialise all singletons (useful for tests / hot-reload)."""
    global _store, _kg, _fewshot, _memory
    _store = FaissStore()
    _kg = KnowledgeGraph()
    _fewshot = FewShotClassifier()
    _memory = Memory()
