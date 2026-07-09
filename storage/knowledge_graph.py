"""Knowledge Graph — data-driven causal association graph.

Accumulates, per defect type, the factory conditions that co-occur with it
(temp / supplier / machine / shift / ...) and maps each defect to a
recommended fix. This is the system's evolving "understanding" of *why*
defects happen, built purely from inspections + human feedback.

No external deps — persisted as JSON.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from config import KG_PATH

# Curated fixes for known defect families. Used until (and alongside) data
# accumulates. Keyed by exact class or by the leading token of the class name.
FALLBACK_FIXES = {
    "crazing": "Reduce cooling rate; inspect mold thermal profile.",
    "inclusion": "Improve melt filtration; audit raw material purity.",
    "patches": "Stabilize coating applicator pressure.",
    "pitted_surface": "Check lubricant contamination; tighten pH control.",
    "scratches": "Inspect conveyor guides; reduce part-contact friction.",
    "rolled-in_scale": "Descale billet before rolling; tune rolling temp.",
    "scuff": "Add buffer between part and fixture; review handling.",
    "bottle_broken_large": "Inspect capping torque; re-align bottle gripper.",
    "bottle_broken_small": "Lower fill pressure; check neck-mold wear.",
    "bottle_contamination": "Sanitize filler line; review CIP cycle.",
    "bottle_cracked": "Reduce thermal shock at the cooling stage.",
    "grid": "Recalibrate deposition-mask alignment.",
    "blowhole": "Degas melt; lower pour temperature.",
    "break": "Inspect support structure; reduce overhang stress.",
    "fray": "Check tensioner; resharpen cutting blade.",
    "glue": "Recalibrate adhesive-dispenser volume.",
    "cut": "Adjust blade depth; verify feed rate.",
    "color": "Recalibrate dye mix ratio.",
    "fold": "Flatten prior to fold; reduce fold speed.",
    "linen": "Tune press-pressure uniformity.",
    "metal": "Remove foreign metal; add inline metal detector.",
    "thread": "Inspect screw gauge; replace worn tap.",
    "scratch": "Polish contact surfaces; clean transport rollers.",
    "screwed": "Verify fastener spec; torque-check tooling.",
    "screw_scratch_neck": "Inspect fastener seating; verify torque spec and neck finish for scratch marks.",
    "rough": "Improve surface-finishing pass parameters.",
    "blemish": "Audit surface-contact media cleanliness.",
    "pinhole": "Increase coating thickness; remove solvent bubbles.",
    "puncture": "Clear path obstructions; inspect puncture probes.",
}

# Numeric metadata fields are bucketed into 10-wide ranges so the graph
# captures "high temp" trends rather than near-unique values.
NUMERIC_FIELDS = {"Temperature", "Humidity"}


class KnowledgeGraph:
    def __init__(self, path: str | Path = KG_PATH):
        self.path = Path(path)
        self.inspections = 0
        self.defect_counts: Counter = Counter()
        self.cond_counts: dict[str, Counter] = {}
        self.load()

    # ---------- ingestion ----------
    def _cond_key(self, field: str, value) -> str:
        if field in NUMERIC_FIELDS and isinstance(value, (int, float)):
            lo = int(value // 10) * 10
            return f"{field}:{lo}-{lo + 10}"
        return f"{field}:{value}"

    def add_inspection(self, defect: str, metadata: dict) -> None:
        self.inspections += 1
        self.defect_counts[defect] += 1
        cc = self.cond_counts.setdefault(defect, Counter())
        for field, value in (metadata or {}).items():
            if field in ("BatchID",):
                continue
            cc[self._cond_key(field, value)] += 1
        self.save()

    def add_edge(self, defect: str, condition: str) -> None:
        # compat shim for the documented contract
        self.cond_counts.setdefault(defect, Counter())[condition] += 1

    def add_rca(self, defect: str, winning_cause: str, actions: list) -> None:
        """Grow the graph from an LLM root-cause analysis: the winning cause
        and each recommended action become new edges for this defect, so the
        knowledge graph accumulates learned root causes over time."""
        if not defect:
            return
        cc = self.cond_counts.setdefault(defect, Counter())
        if winning_cause:
            cc[f"cause:{winning_cause}"] += 1
        for a in (actions or []):
            if a:
                cc[f"fix:{a}"] += 1
        self.save()

    # ---------- query ----------
    def get_causes(self, defect: str, top_k: int = 4) -> list:
        cc = self.cond_counts.get(defect)
        if not cc:
            return []
        # keep real factory conditions only — learned cause:/fix: edges are
        # surfaced via get_fix(), not as "associated conditions"
        real = [(k, c) for k, c in cc.items()
                if not (k.startswith("cause:") or k.startswith("fix:"))]
        if not real:
            return []
        total = sum(c for _, c in real)
        n = self.defect_counts.get(defect, 0)
        real.sort(key=lambda x: -x[1])
        return [
            {"condition": k, "count": c, "share": c / total, "n": n}
            for k, c in real[:top_k]
        ]

    def get_fix(self, defect: str) -> str:
        # learned fixes (from Teach / multi-agent RCA) take priority over the
        # static fallback table, so the graph visibly accumulates experience
        cc = self.cond_counts.get(defect)
        if cc:
            fixes = [(k, c) for k, c in cc.items() if k.startswith("fix:")]
            if fixes:
                fixes.sort(key=lambda x: -x[1])
                return fixes[0][0][len("fix:"):]
        if defect in FALLBACK_FIXES:
            return FALLBACK_FIXES[defect]
        lead = defect.split("_")[0]
        if lead in FALLBACK_FIXES:
            return FALLBACK_FIXES[lead]
        return "No fix mapped yet — add one via Teach."

    def top_defects(self, top_k: int = 5) -> list:
        return self.defect_counts.most_common(top_k)

    def summary(self) -> dict:
        return {
            "inspections": self.inspections,
            "distinct_defects": len(self.defect_counts),
            "top_defects": self.top_defects(),
        }

    # ---------- persistence ----------
    def as_dict(self) -> dict:
        return {
            "inspections": self.inspections,
            "defect_counts": dict(self.defect_counts),
            "cond_counts": {k: dict(v) for k, v in self.cond_counts.items()},
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.json")
        with open(tmp, "w") as f:
            json.dump(self.as_dict(), f, indent=2)
        tmp.replace(self.path)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            d = json.load(open(self.path))
            self.inspections = d.get("inspections", 0)
            self.defect_counts = Counter(d.get("defect_counts", {}))
            self.cond_counts = {k: Counter(v) for k, v in d.get("cond_counts", {}).items()}
        except Exception:
            pass
