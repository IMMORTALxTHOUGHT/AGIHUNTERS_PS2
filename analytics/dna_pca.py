"""STAGE 9 - Defect DNA (PCA over factory-condition space).

Treats each recorded inspection as a point in factory-condition space
(Temperature, Humidity, Pressure, MachineAge, LubricationHours) and projects
it to 2D with PCA. Defects that share causes cluster together — the
"DNA" of each defect family. Color = defect type; X = centroid.

PCA is implemented in pure numpy (no sklearn dependency). The plotting
function lazy-imports matplotlib so the compute path works headless.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

NUMERIC_FIELDS = ["Temperature", "Humidity", "Pressure", "MachineAge", "LubricationHours"]


def _feature_matrix(cases: list) -> tuple:
    X, labels, meta = [], [], []
    for c in cases:
        m = c.get("metadata") or {}
        row = [float(m.get(f, 0.0) or 0.0) for f in NUMERIC_FIELDS]
        if sum(row) == 0:  # no usable metadata
            continue
        X.append(row)
        labels.append(c.get("defect_type", "unknown"))
        meta.append(c)
    return np.array(X, dtype=float), labels, meta


def project(X: np.ndarray) -> tuple:
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2)), [1.0, 0.0]
    Xc = X - X.mean(0)
    cov = np.cov(Xc, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    comps = vecs[:, order[:2]]
    var_exp = (vals[order][:2] / vals.sum()).tolist()
    return Xc @ comps, var_exp


def compute_dna(cases: list) -> dict:
    X, labels, meta = _feature_matrix(cases)
    if X.shape[0] == 0:
        return {"points": [], "centroids": {}, "variance_explained": [], "n": 0}
    coords, var_exp = project(X)
    points = [
        {"x": float(x), "y": float(y), "defect": lab,
         "anomaly": float(c.get("anomaly_score", 0.0)),
         "novel": bool(c.get("is_novel", False))}
        for (x, y), lab, c in zip(coords, labels, meta)
    ]
    groups = defaultdict(list)
    for p in points:
        groups[p["defect"]].append((p["x"], p["y"]))
    centroids = {
        k: [float(np.mean([g[0] for g in v])), float(np.mean([g[1] for g in v]))]
        for k, v in groups.items()
    }
    return {"points": points, "centroids": centroids,
            "variance_explained": var_exp, "n": len(points)}


def render_dna_figure(cases: list, path: str, dpi: int = 90) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    data = compute_dna(cases)
    if not data["points"]:
        return None

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    defects = sorted({p["defect"] for p in data["points"]})
    cmap = plt.get_cmap("tab10")
    for i, d in enumerate(defects):
        xs = [p["x"] for p in data["points"] if p["defect"] == d]
        ys = [p["y"] for p in data["points"] if p["defect"] == d]
        ax.scatter(xs, ys, label=d, s=30, color=cmap(i % 10),
                   alpha=0.8, edgecolor="white", linewidth=0.4)
        cx, cy = data["centroids"].get(d, (0, 0))
        ax.scatter([cx], [cy], marker="X", s=110, color=cmap(i % 10),
                   edgecolor="black", linewidth=0.8, zorder=5)
    ve = data["variance_explained"]
    ax.set_xlabel(f"PC1 ({ve[0] * 100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({ve[1] * 100:.0f}% var)")
    ax.set_title("Defect-DNA (factory-condition PCA)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
