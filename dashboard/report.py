"""Full inspection report -> PDF.

Builds a self-contained, downloadable PDF for a single inspected part that
walks the whole pipeline end to end: part image + anomaly heatmap,
classification, factory metadata, similar past cases, knowledge-graph fix,
multi-agent root-cause analysis, and the factory analytics (health,
defects-by-type, by machine/shift, Defect-DNA, calibration).

Uses matplotlib's PdfPages only, so it adds NO new dependency (matplotlib is
already required by the analytics figures). If matplotlib is unavailable the
builder returns None and the dashboard button simply does nothing.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime

from storage import database
from storage.knowledge_graph import KnowledgeGraph
from analytics.health import health
from analytics.dna_pca import render_dna_figure
from analytics.calibration import render_calibration_figure

_W = 8.27   # A4 width  (inches)
_H = 11.69  # A4 height (inches)


def _sev_text(result: dict) -> str:
    if result.get("is_novel_defect"):
        return "Needs review (novel / low-confidence)"
    score = float(result.get("anomaly_score", 0.0) or 0.0)
    conf = float(result.get("vit_confidence", 1.0) or 1.0)
    if score >= 1.0 or conf < 0.55:
        return "High"
    if score >= 0.4 or conf < 0.80:
        return "Mid"
    return "Low"


class _Report:
    """Minimal flowing-page writer on top of matplotlib PdfPages."""

    def __init__(self, path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        self.plt = plt
        self.pdf = PdfPages(path)
        self.path = path
        self.fig = None
        self.y = 0.0
        self._new_page()

    def _fx(self, ix: float) -> float:
        return ix / _W

    def _fy(self, iy: float) -> float:  # iy measured down from the top
        return iy / _H

    def _new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            self.plt.close(self.fig)
        self.fig = self.plt.figure(figsize=(_W, _H))
        self.fig.patch.set_facecolor("white")
        self.y = _H - 0.7

    def _ensure(self, need: float):
        if self.y - need < 0.6:
            self._new_page()

    def title_block(self, text, sub=None):
        self.fig.text(self._fx(0.6), self._fy(self.y), text, fontsize=20,
                      fontweight="bold", color="#0b3d1e", va="top")
        self.y -= 0.38
        if sub:
            self.fig.text(self._fx(0.6), self._fy(self.y), sub, fontsize=10,
                          color="#6b7280", va="top")
            self.y -= 0.28
        self.y -= 0.05

    def heading(self, text, size=14):
        self._ensure(0.5)
        self.fig.text(self._fx(0.6), self._fy(self.y), text, fontsize=size,
                      fontweight="bold", color="#0b3d1e", va="top")
        self.y -= 0.40

    def line(self, text, size=10, color="#111827", bold=False):
        self._ensure(0.22)
        self.fig.text(self._fx(0.6), self._fy(self.y), text, fontsize=size,
                      color=color, fontweight="bold" if bold else "normal",
                      va="top")
        self.y -= 0.20

    def gap(self, dy=0.14):
        self.y -= dy

    def images(self, items, max_w=3.6, max_h=3.6):
        if not items:
            return
        n = len(items)
        cell_w = (_W - 1.2) / n
        w = min(max_w, cell_w - 0.2)
        h = min(max_h, 3.6)
        for i, (p, title) in enumerate(items):
            left = 0.6 + i * cell_w
            bottom = self.y - h
            ax = self.fig.add_axes([self._fx(left), self._fy(bottom) - 0,
                                    self._fx(w), self._fy(h)])
            try:
                img = self.plt.imread(p) if isinstance(p, str) else p
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "image unavailable", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            for s in ax.spines.values():
                s.set_visible(False)
        self.y -= (h + 0.35)

    def table(self, headers, rows):
        if not rows:
            self.line("(no data yet)", color="#6b7280")
            return
        ncol = len(headers)
        cell_h = 0.22
        total_h = cell_h * (len(rows) + 1)
        self._ensure(total_h + 0.12)
        x0 = 0.6
        width = _W - 1.2
        bottom = self.y - total_h
        ax = self.fig.add_axes([self._fx(x0), self._fy(bottom),
                                self._fx(width), self._fy(total_h)])
        tbl = ax.table(cellText=rows, colLabels=headers, loc="upper left",
                       cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.7)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.4)
            if r == 0:
                cell.set_facecolor("#0b3d1e")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f3f4f6")
        ax.axis("off")
        self.y -= (total_h + 0.18)

    def close(self):
        self.pdf.savefig(self.fig)
        self.plt.close(self.fig)
        self.pdf.close()


def build_pdf(image_path: str, result: dict, rca: dict | None,
              out_dir: str) -> str | None:
    try:
        import cv2
    except Exception:
        cv2 = None

    os.makedirs(out_dir, exist_ok=True)
    td = tempfile.mkdtemp(prefix="fm_rpt_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"forgemind_report_{name}_{stamp}.pdf")

    kg = KnowledgeGraph()
    fix_for = lambda lab: kg.get_fix(lab or "")

    # --- prepare image assets in a temp dir ---
    items = [(image_path, "Part image")]
    overlay = result.get("heatmap_overlay")
    if overlay is not None and cv2 is not None:
        tmp_ov = os.path.join(td, "overlay.png")
        cv2.imwrite(tmp_ov, overlay)
        items.append((tmp_ov, "Anomaly heatmap"))

    conn = database.get_connection()
    cases = database.get_cases(conn)
    conn.close()
    dna_p = render_dna_figure(cases, os.path.join(td, "dna.png"))
    cal_p = render_calibration_figure(cases, os.path.join(td, "cal.png"))
    h = health(cases)

    rep = _Report(out_path)
    rep.title_block(
        "ForgeMind \u2014 Defect Inspection Report",
        f"Part: {os.path.basename(image_path)}    Generated: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}")

    # 1. classification
    rep.heading("1 \u00b7 Classification")
    rep.images(items)
    rep.line(f"Defect type : {result.get('defect', '?')}", bold=True)
    rep.line(f"Confidence  : {min(float(result.get('vit_confidence', 0) or 0) * 100, 99.7):.1f}%")
    rep.line(f"Severity    : {_sev_text(result)}")
    rep.line(f"Anomaly score: {float(result.get('anomaly_score', 0) or 0):.3f}")
    rep.line(f"Novel / unseen defect: {'Yes' if result.get('is_novel_defect') else 'No'}")
    rep.gap()

    # 2. factory metadata
    rep.heading("2 \u00b7 Factory metadata")
    meta = result.get("metadata", {}) or {}
    rep.table(["field", "value"], [[str(k), str(v)] for k, v in meta.items()])

    # 3. similar past cases
    rep.heading("3 \u00b7 Similar past cases")
    sim_rows = [[str(i), r.get("label", ""), f'{r.get("similarity", 0):.3f}',
                 fix_for(r.get("label", ""))]
                for i, r in enumerate(result.get("similar_cases", [])[:5], 1)]
    rep.table(["case", "defect", "similarity", "resolution / fix"], sim_rows)

    # 4. knowledge graph
    rep.heading("4 \u00b7 Knowledge graph & memory")
    know = kg.get_fix(result.get("defect", ""))
    summ = result.get("defect", "unknown")
    rep.line(f"Recommended fix for '{summ}': {know}", bold=True)
    rep.line(f"Recorded {len(cases)} inspections across the factory so far.")
    rep.gap()

    # 5. multi-agent root-cause analysis
    rep.heading("5 \u00b7 AI root-cause analysis (multi-agent)")
    if rca:
        votes = rca.get("votes") or []
        vrows = [[v.get("role", ""), v.get("cause", ""),
                  f'{min(float(v.get("conf", 0) or 0) * 100, 99.7):.1f}%']
                 for v in votes]
        rep.table(["specialist", "hypothesis", "conf"], vrows)
        winning = rca.get("winning_cause") or "\u2014"
        rep.line(f"Winning root cause: {winning}",
                 bold=True, color="#0b3d1e")
        rep.line(f"Rationale: {rca.get('rationale', '')}")
        actions = rca.get("actions") or []
        if actions:
            rep.line("Recommended actions:", bold=True)
            for a in actions:
                rep.line(f"   \u2022 {a}")
    else:
        rep.line("(Run 'Explain root cause' to attach the analysis.)",
                 color="#6b7280")
    rep.gap()

    # 6. factory analytics
    rep.heading("6 \u00b7 Factory analytics (all inspections)")
    fp = min(h["factory_pct"], 99.7)
    tier = h["tier"]
    meaning = ("The line is running clean \u2014 most parts pass."
               if tier == "Good"
               else "Defect rate is elevated \u2014 keep an eye on it."
               if tier == "Watch"
               else "Defect rate is high \u2014 investigate now.")
    rep.line(f"Factory health: {fp:.0f}%  (tier: {tier}) \u2014 {meaning}",
             bold=True, color="#0b3d1e")
    ai_conf = max(0.0, 100.0 - h["novel_rate"])
    rep.line(f"Parts inspected: {h['n']}   |   Defect rate: "
             f"{min(h['defect_rate'], 99.7):.0f}%   |   Uncertain/novel: "
             f"{min(h['novel_rate'], 99.7):.0f}%   |   AI confident: "
             f"{min(ai_conf, 99.7):.0f}%")

    dist = Counter(c.get("defect_type") or c.get("defect") or "unknown"
                   for c in cases)
    n = max(1, h["n"])
    dt_rows = [[k, str(v), f"{v / n * 100:.0f}%"] for k, v in dist.most_common()]
    rep.table(["defect type", "count", "share"], dt_rows)

    mach_rows = [[k, f"{v:.0f}%"] for k, v in sorted(h["by_machine"].items())]
    rep.table(["machine", "health"], mach_rows)
    shift_rows = [[k, f"{v:.0f}%"] for k, v in sorted(h["by_shift"].items())]
    rep.table(["shift", "health"], shift_rows)

    extra = []
    if dna_p:
        extra.append((dna_p, "Defect-DNA (conditions vs defect type)"))
    if cal_p:
        extra.append((cal_p, "Confidence calibration"))
    if extra:
        rep.images(extra, max_w=3.6, max_h=3.2)

    rep.close()
    shutil.rmtree(td, ignore_errors=True)
    return out_path


if __name__ == "__main__":
    print("build_pdf needs a real inspection result; import from dashboard.app.")
