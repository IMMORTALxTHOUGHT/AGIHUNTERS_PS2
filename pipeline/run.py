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
from models.embedder import embed
from storage.faiss_store import FaissStore
from storage.knowledge_graph import KnowledgeGraph
from agents.debate import run_debate
from agents.moderator import moderate
from analytics.dna_pca import project
from analytics.calibration import calibrate
from analytics.health import health


def run(image_path: str) -> dict:
    # 1. Upload
    img_pil, img_np, x_vit = load_image(image_path)
    # 2. Synthetic metadata
    meta = make_metadata(image_path)
    # 3. PatchCore detector
    pc = PatchCore()
    score, heat, roi, crop = pc.score(x_vit)
    # 4. ViT classifier
    defect, conf = classify(x_vit)
    # 5. Embedder
    emb = embed(x_vit)
    # 6. FAISS retrieval
    store = FaissStore()
    cases = store.search(emb, k=5)
    # 7. Knowledge graph
    kg = KnowledgeGraph()
    chains = kg.chains_from(defect)
    # 8. Multi-agent debate + moderator
    votes = run_debate({"defect": defect, "heat": heat.shape},
                       cases, meta, chains)
    rca = moderate(votes)
    # 9. Defect DNA
    dna = project(emb.reshape(1, -1))
    # 10. Calibration
    cal = calibrate(conf, np.array([0.1, 0.2]), votes, meta)
    # 11. Factory health
    h = health([{"meta": meta, "is_defect": score > 0.5}])
    # 12. Self-learning (TODO: store.add / kg.add_edge / sqlite insert)
    # 13. Recommendation (inside rca["actions"])
    # 14. Dashboard consumes the returned dict
    return {
        "anomaly_score": score,
        "defect": defect,
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
