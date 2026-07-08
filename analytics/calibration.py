"""STAGE 10 - Confidence Calibration (decomposed sub-scores).

Contract:
  calibrate(class_conf, faiss_D, votes, metadata)
      -> {"visual","history","metadata","consensus","overall"}
"""
from __future__ import annotations
import numpy as np


def calibrate(class_conf: float, faiss_D, votes: list, metadata: dict) -> dict:
    faiss_D = np.asarray(faiss_D, dtype=float)
    history = 1.0 - float(faiss_D.mean() / (faiss_D.max() + 1e-9)) if faiss_D.size else 0.5
    consensus = 1.0 - float(np.std([v["conf"] for v in votes])) if votes else 0.5
    meta = 0.9 if metadata.get("Shift") == "A" else 0.7
    overall = 0.3 * class_conf + 0.3 * history + 0.2 * meta + 0.2 * consensus
    return {
        "visual": round(class_conf, 3),
        "history": round(history, 3),
        "metadata": round(meta, 3),
        "consensus": round(consensus, 3),
        "overall": round(overall, 3),
    }
