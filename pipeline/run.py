"""Orchestrator - runs stages 1..14 in order.

Run from the repo root:
    python -m pipeline.run --image <path-to-image>

The scaffold stubs return correctly-shaped DUMMY data, so this runs
end-to-end anywhere (no GPU / no models needed) to demonstrate the wiring.
Replace each module's dummy body with the real implementation on the box.

Contract: run(image_path) -> dict (the full analysis result)
"""
from __future__ import annotations
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
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


VIT_CONF_THRESHOLD = 0.7


def run(image_path: str, human_feedback: dict | None = None) -> dict:
    # Shared state (singletons initialized once; here for scaffold simplicity)
    store = FaissStore()
    kg = KnowledgeGraph()
    fewshot = FewShotClassifier()
    memory = Memory()

    # 1. Upload
    img_pil, img_np, x_vit = load_image(image_path)
    # 2. Synthetic metadata
    meta = make_metadata(image_path)
    # 3. PatchCore detector
    pc = PatchCore()
    score, heat, roi, crop = pc.score(x_vit)
    # 4. ViT classifier
    defect, conf = classify(x_vit)
    # 4b. Few-shot fallback when ViT confidence is low
    is_novel = False
    fewshot_distances = {}
    if conf < VIT_CONF_THRESHOLD:
        emb_for_fewshot = embed(x_vit)
        defect, fs_conf, is_novel, fewshot_distances = fewshot.classify(
            emb_for_fewshot, conf, VIT_CONF_THRESHOLD
        )
        conf = fs_conf if fs_conf > conf else conf
    # 5. Embedder
    emb = embed(x_vit)
    # 6. FAISS retrieval
    cases = store.search(emb, k=5)
    # 7. Knowledge graph
    chains = kg.chains_from(defect)
    # 8. Multi-agent debate + moderator
    votes = run_debate({"defect": defect, "heat": heat.shape,
                        "is_novel": is_novel},
                       cases, meta, chains)
    rca = moderate(votes)
    # 9. Defect DNA
    dna = project(emb.reshape(1, -1))
    # 10. Calibration
    cal = calibrate(conf, np.array([0.1, 0.2]), votes, meta)
    # 11. Factory health
    h = health([{"meta": meta, "is_defect": score > 0.5}])
    # 12. Self-learning memory (mandatory)
    record = {
        "defect_type": defect,
        "metadata": meta,
        "rca": rca,
        "anomaly_score": score,
        "is_novel": is_novel,
    }
    memory.update(emb, record, fewshot=fewshot, human_feedback=human_feedback)
    # 13. Recommendation (inside rca["actions"])
    # 14. Dashboard consumes the returned dict
    return {
        "anomaly_score": score,
        "defect": defect,
        "vit_confidence": conf,
        "fewshot_fallback": conf < VIT_CONF_THRESHOLD,
        "is_novel_defect": is_novel,
        "fewshot_distances": fewshot_distances,
        "metadata": meta,
        "rca": rca,
        "dna": dna.tolist(),
        "calibration": cal,
        "health": h,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to an image")
    args = ap.parse_args()
    print(json.dumps(run(args.image), indent=2, default=str))
