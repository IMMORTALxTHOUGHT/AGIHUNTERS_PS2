"""STAGE 10 - Confidence Calibration summary.

ViT emits a confidence per prediction. A well-calibrated model should be
*uncertain* (low confidence) exactly when it is wrong / novel. We don't always
have ground truth, so we use the novel-flag (conf < VIT_CONF_THRESHOLD) as a
proxy for "should be uncertain" and check the confidence distribution against
it: low-confidence bins should be dominated by novel cases, high-confidence
bins should have almost none. This surfaces mis-calibration at a glance.
"""
from __future__ import annotations

import numpy as np

from config import VIT_CONF_THRESHOLD

BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]


def calibration_summary(cases: list, conf_threshold: float = VIT_CONF_THRESHOLD) -> dict:
    confs = [float(c.get("vit_confidence") or 0.0) for c in cases
             if c.get("vit_confidence") is not None]
    if not confs:
        return {"n": 0, "mean_conf": 0.0, "novel_rate": 0.0,
                "threshold": conf_threshold, "bins": []}

    novel = [bool(c.get("is_novel", False)) for c in cases
             if c.get("vit_confidence") is not None]
    confs = np.array(confs)
    novel = np.array(novel)

    bins = []
    for lo, hi in BINS:
        mask = (confs >= lo) & (confs < hi)
        cnt = int(mask.sum())
        frac_novel = float(novel[mask].mean()) if cnt else 0.0
        bins.append({
            "range": f"{lo:.2f}-{min(hi, 1.0):.2f}",
            "count": cnt,
            "novel_rate": round(frac_novel, 3),
        })

    return {
        "n": len(confs),
        "mean_conf": round(float(confs.mean()), 3),
        "novel_rate": round(float(novel.mean()), 3),
        "threshold": conf_threshold,
        "bins": bins,
    }


def render_calibration_figure(cases: list, path: str, dpi: int = 90) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    summary = calibration_summary(cases)
    bins = summary["bins"]
    if not bins:
        return None

    ranges = [b["range"] for b in bins]
    counts = [b["count"] for b in bins]
    novel_rates = [b["novel_rate"] * 100 for b in bins]

    fig, ax1 = plt.subplots(figsize=(5.4, 3.8))
    ax1.bar(ranges, counts, color="#3b82f6", alpha=0.75, label="cases")
    ax1.set_ylabel("cases", color="#3b82f6")
    ax1.set_xlabel("ViT confidence bin")
    ax2 = ax1.twinx()
    ax2.plot(ranges, novel_rates, "o-", color="#ff6a33", label="% novel")
    ax2.set_ylabel("% novel-flagged", color="#ff6a33")
    ax2.set_ylim(0, 105)
    ax1.set_title("Confidence calibration (low conf → high % novel expected)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
