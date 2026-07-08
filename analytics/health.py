"""STAGE 11 - Factory Health Score (aggregate risk from inspection history).

Reads accumulated cases and reports a factory health percentage, per-machine
and per-shift breakdowns, and a Good / Watch / Critical tier. Defectiveness is
percentile-based on anomaly score plus any novel-flagged inspection, so it is
robust to the uncalibrated raw PatchCore distance scale.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def health(cases: list) -> dict:
    if not cases:
        return {"factory_pct": 100.0, "tier": "Good", "n": 0,
                "defect_rate": 0.0, "novel_rate": 0.0,
                "by_machine": {}, "by_shift": {}}

    scores = np.array([float(c.get("anomaly_score", 0.0) or 0.0) for c in cases])
    thr = float(np.percentile(scores, 80)) if len(scores) > 1 else float(scores.max())
    flags = [
        (float(c.get("anomaly_score", 0.0) or 0.0) >= thr) or bool(c.get("is_novel", False))
        for c in cases
    ]
    n = len(cases)
    nd = sum(flags)
    rate = nd / n
    factory_pct = round(100.0 * (1 - rate), 1)

    by_machine, by_shift = defaultdict(list), defaultdict(list)
    for c, f in zip(cases, flags):
        m = (c.get("metadata") or {}).get("Machine", "?")
        s = (c.get("metadata") or {}).get("Shift", "?")
        by_machine[m].append(f)
        by_shift[s].append(f)

    def agg(g):
        return {k: round(100.0 * (1 - sum(v) / len(v)), 1) for k, v in g.items()}

    tier = "Good" if factory_pct >= 80 else ("Watch" if factory_pct >= 60 else "Critical")
    novel_rate = round(100.0 * sum(1 for c in cases if c.get("is_novel")) / n, 1)

    return {"factory_pct": factory_pct, "tier": tier, "n": n,
            "defect_rate": round(100.0 * rate, 1), "novel_rate": novel_rate,
            "by_machine": agg(by_machine), "by_shift": agg(by_shift)}
