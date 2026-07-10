"""Full inspection report -> PDF.

Builds a self-contained, downloadable PDF for a single inspected part that
walks the whole pipeline end to end: part image + anomaly heatmap,
classification, factory metadata, similar past cases, knowledge-graph fix,
multi-agent root-cause analysis, and the factory analytics (health,
defects-by-type, by machine/shift, Defect-DNA, calibration).

Uses matplotlib's PdfPages only, so it adds NO new dependency (matplotlib is
already required by the analytics figures). If matplotlib is unavailable the
builder returns None and the dashboard button simply does nothing.

Layout note: all text flows on a shared vertical cursor and tables are drawn
as monospaced, column-aligned text lines (not matplotlib axes tables) so
nothing can overlap.
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
_LEFT = 0.6
_USABLE = _W - 2 * _LEFT


def _sev_text(result: dict) -> str:
    if result.get("is_ood"):
        return "Out of distribution (unrecognized part)"
    if result.get("is_novel_defect"):
        return "Needs review (novel / low-confidence)"
    score = float(result.get("anomaly_score", 0.0) or 0.0)
    conf = float(result.get("vit_confidence", 1.0) or 1.0)
    if score >= 1.0 or conf < 0.55:
        return "High"
    if score >= 0.4 or conf < 0.80:
        return "Mid"
    return "Low"


def _wrap(text: str, width: int):
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


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

    def _fy(self, iy_from_top: float) -> float:
        return iy_from_top / _H

    def _new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            self.plt.close(self.fig)
        self.fig = self.plt.figure(figsize=(_W, _H))
        self.fig.patch.set_facecolor("white")
        self.y = _H - 0.7

    def _ensure(self, need: float):
        if self.y - need < 0.55:
            self._new_page()

    def _dy(self, size: float) -> float:
        return max(0.17, size / 72.0 * 1.4)

    def title_block(self, text, sub=None):
        self.fig.text(self._fx(_LEFT), self._fy(self.y), text, fontsize=20,
                      fontweight="bold", color="#0b3d1e", va="top")
        self.y -= self._dy(20)
        if sub:
            for ln in _wrap(sub, 95):
                self.fig.text(self._fx(_LEFT), self._fy(self.y), ln,
                              fontsize=10, color="#6b7280", va="top")
                self.y -= self._dy(10)
        self.y -= 0.06

    def heading(self, text, size=13):
        self._ensure(0.5)
        self.fig.text(self._fx(_LEFT), self._fy(self.y), text, fontsize=size,
                      fontweight="bold", color="#0b3d1e", va="top")
        self.y -= self._dy(size) + 0.10

    def line(self, text, size=10, color="#111827", bold=False, mono=False):
        width = int(_USABLE / (size / 72.0 * 0.60)) if not mono else 100
        width = min(width, 104)
        dy = self._dy(size)
        for ln in _wrap(text, width):
            self._ensure(dy)
            self.fig.text(self._fx(_LEFT), self._fy(self.y), ln, fontsize=size,
                          color=color, fontweight="bold" if bold else "normal",
                          family="monospace" if mono else "DejaVu Sans",
                          va="top")
            self.y -= dy

    def gap(self, dy=0.12):
        self.y -= dy

    def images(self, items, max_w=3.4, max_h=3.2):
        if not items:
            return
        n = len(items)
        cell_w = (_W - 2 * _LEFT) / n
        w = min(max_w, cell_w - 0.2)
        h = min(max_h, 3.4)
        for i, (p, title) in enumerate(items):
            left = _LEFT + i * cell_w
            bottom = self.y - h
            ax = self.fig.add_axes([self._fx(left), self._fy(bottom),
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
        self.y -= (h + 0.30)

    def table(self, headers, rows):
        if not rows:
            self.line("(no data yet)", color="#6b7280")
            return
        cols = list(zip(*([headers] + rows)))
        widths = [max(len(str(c)) for c in col) for col in cols]
        widths = [min(x, 22) for x in widths]
        total = sum(widths) + 2 * len(widths)

        def fmt(row):
            return "  ".join(str(c)[:22].ljust(widths[i])
                            for i, c in enumerate(row))

        self.line(fmt(headers), size=8.5, mono=True, bold=True, color="#0b3d1e")
        self.line("-" * min(total, 100), size=8.5, mono=True, color="#9ca3af")
        for r in rows:
            self.line(fmt(r), size=8.5, mono=True)

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
        "ForgeMind - Defect Inspection Report",
        f"Part: {os.path.basename(image_path)}    Generated: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}")

    # 1. classification
    rep.heading("1 . Classification")
    rep.images(items)
    if result.get("is_ood"):
        rep.line("Defect type        : UNRECOGNIZED — out of distribution",
                 bold=True)
        rep.line(f"Nearest-match conf : "
                 f"{min(float(result.get('vit_confidence', 0) or 0) * 100, 99.7):.1f}%")
    else:
        rep.line(f"Defect type        : {result.get('defect', '?')}", bold=True)
        rep.line(f"Confidence         : "
                 f"{min(float(result.get('vit_confidence', 0) or 0) * 100, 99.7):.1f}%")
    rep.line(f"Severity           : {_sev_text(result)}")
    rep.line(f"Anomaly score      : {float(result.get('anomaly_score', 0) or 0):.3f}")
    rep.line(f"Novel / unseen     : {'Yes' if result.get('is_novel_defect') else 'No'}")
    rep.gap()

    # 2. factory metadata
    rep.heading("2 . Factory metadata")
    meta = result.get("metadata", {}) or {}
    rep.table(["field", "value"],
              [[str(k), str(v)[:30]] for k, v in meta.items()])

    # 3. similar past cases
    rep.heading("3 . Similar past cases")
    rep.line("Top visual neighbours from the FAISS memory (self-match excluded).",
             size=9)
    sim_rows = [[str(i), str(r.get("label", ""))[:30],
                  f'{r.get("similarity", 0):.3f}', str(fix_for(r.get("label", "")))[:30]]
                 for i, r in enumerate(result.get("similar_cases", [])[:5], 1)]
    rep.table(["case", "defect", "similarity", "resolution / fix"], sim_rows)

    # 4. knowledge graph
    rep.heading("4 . Knowledge graph & memory")
    defect = result.get("defect", "unknown")
    fix = kg.get_fix(defect)
    rep.line(f"Recommended fix for '{defect}': {fix}", bold=True)
    causes = kg.get_causes(defect)
    if causes:
        n = causes[0].get("n", 0)
        rep.line(f"Associated conditions for '{defect}' (from {n} inspection(s)):",
                 bold=True)
        for c in causes:
            low = "  [low support]" if c["count"] < 3 else ""
            rep.line(f"  - {c['condition']}: seen {c['count']} "
                     f"({c['share'] * 100:.0f}%){low}", size=9)
    else:
        rep.line(f"Not enough data yet for '{defect}' — inspect more parts.")
    rep.line(f"Recorded {len(cases)} inspections across the factory so far.")
    rep.gap()

    # 5. multi-agent root-cause analysis
    rep.heading("5 . AI root-cause analysis (multi-agent)")
    if rca:
        votes = rca.get("votes") or []
        vrows = [[str(v.get("role", ""))[:30], str(v.get("cause", ""))[:30],
                  f'{min(float(v.get("conf", 0) or 0) * 100, 99.7):.1f}%']
                 for v in votes]
        rep.table(["specialist", "hypothesis", "conf"], vrows)
        rep.line(f"Winning root cause : {rca.get('winning_cause') or '-'}",
                 bold=True, color="#0b3d1e")
        rep.line(f"Rationale : {rca.get('rationale', '')}")
        actions = rca.get("actions") or []
        if actions:
            rep.line("Recommended actions:", bold=True)
            for a in actions:
                rep.line(f"   - {a}")
    else:
        rep.line("(Run 'Explain root cause' to attach the analysis.)",
                 color="#6b7280")
    rep.gap()

    # 6. factory analytics
    rep.heading("6 . Factory analytics (all inspections)")
    fp = min(h["factory_pct"], 99.7)
    tier = h["tier"]
    meaning = ("The line is running clean - most parts pass."
               if tier == "Good"
               else "Defect rate is elevated - keep an eye on it."
               if tier == "Watch"
               else "Defect rate is high - investigate now.")
    rep.line(f"Factory health: {fp:.0f}%  (tier: {tier}) - {meaning}",
             bold=True, color="#0b3d1e")
    ai_conf = max(0.0, 100.0 - h["novel_rate"])
    rep.line(f"Parts inspected: {h['n']}   |   Defect rate: "
             f"{min(h['defect_rate'], 99.7):.0f}%   |   Uncertain/novel: "
             f"{min(h['novel_rate'], 99.7):.0f}%   |   AI confident: "
             f"{min(ai_conf, 99.7):.0f}%")

    dist = Counter(c.get("defect_type") or c.get("defect") or "unknown"
                   for c in cases)
    n = max(1, h["n"])
    dt_rows = [[str(k)[:30], str(v), f"{v / n * 100:.0f}%"]
               for k, v in dist.most_common()]
    rep.table(["defect type", "count", "share"], dt_rows)

    mach_rows = [[str(k)[:30], f"{v:.0f}%"] for k, v in sorted(h["by_machine"].items())]
    rep.table(["machine", "health"], mach_rows)
    shift_rows = [[str(k)[:30], f"{v:.0f}%"] for k, v in sorted(h["by_shift"].items())]
    rep.table(["shift", "health"], shift_rows)

    extra = []
    if dna_p:
        extra.append((dna_p, "Defect-DNA (conditions vs defect type)"))
    if cal_p:
        extra.append((cal_p, "Confidence calibration"))
    if extra:
        rep.images(extra, max_w=3.4, max_h=3.0)

    rep.close()
    shutil.rmtree(td, ignore_errors=True)
    return out_path


if __name__ == "__main__":
    print("build_pdf needs a real inspection result; import from dashboard.app.")


def build_batch_pdf(analysis: dict, out_dir: str) -> str | None:
    """Batch root-cause report: one section per defect type (members table +
    multi-agent debate + winning cause + actions), an unrecognized-parts
    section, and the factory analytics tail. Consumes the dict produced by
    dashboard.app.batch_analyze()."""
    os.makedirs(out_dir, exist_ok=True)
    td = tempfile.mkdtemp(prefix="fm_brpt_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"forgemind_batch_report_{stamp}.pdf")

    conn = database.get_connection()
    cases = database.get_cases(conn)
    conn.close()
    dna_p = render_dna_figure(cases, os.path.join(td, "dna.png"))
    cal_p = render_calibration_figure(cases, os.path.join(td, "cal.png"))
    h = health(cases)

    rep = _Report(out_path)
    rep.title_block("ForgeMind - Batch Root-Cause Report",
                    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")

    rep.heading("Batch summary")
    rep.line(f"Parts analyzed : {analysis.get('total', 0)}")
    rep.line(f"Defective      : {analysis.get('defective', 0)}")
    rep.line(f"Defect types   : {len(analysis.get('groups', []))}")
    rep.line(f"Unrecognized   : {len(analysis.get('unrecognized', []))} "
             f"(out of distribution)")
    rep.gap()

    groups = analysis.get("groups", [])
    if not groups:
        rep.line("No defective parts found in this batch "
                 "(all parts were good or out of distribution).", color="#6b7280")
    for gi, g in enumerate(groups, 1):
        rep.heading(f"{gi}. {g['defect']}  ({g['count']} part(s))")
        s = g["summary"]
        sev = s["severity_dist"]
        sev_txt = ", ".join(f"{k}:{v}" for k, v in sev.items() if v)
        cond = s.get("common_conditions", {})
        cond_txt = ", ".join(f"{k}={v}" for k, v in cond.items()) or "no shared metadata"
        rep.line(f"severity mix [{sev_txt}]   avg anomaly {s['avg_anomaly']:.3f}", size=9)
        rep.line(f"shared conditions: {cond_txt}", size=9)
        rep.gap(0.06)
        mrows = [[m["file"], f'{float(m["conf"]) * 100:.1f}%', str(m["severity"]),
                  f'{float(m["anomaly"]):.3f}'] for m in g["members"][:30]]
        rep.table(["file", "conf", "severity", "anomaly"], mrows)
        rep.heading("Multi-agent debate (common root cause)", size=11)
        vrows = [[str(v.get("role", "")), str(v.get("cause", "")),
                  f'{min(float(v.get("conf", 0) or 0) * 100, 99.7):.1f}%']
                 for v in g.get("votes", [])]
        rep.table(["specialist", "hypothesis", "conf"], vrows)
        v = g.get("verdict") or {}
        rep.line(f"Winning root cause: {v.get('winning_cause', '-')}",
                 bold=True, color="#0b3d1e")
        rep.line(f"Rationale: {v.get('rationale', '')}")
        actions = v.get("actions") or []
        if actions:
            rep.line("Recommended actions:", bold=True)
            for a in actions:
                rep.line(f"   - {a}")
        rep.gap(0.12)

    unrec = analysis.get("unrecognized", [])
    if unrec:
        rep.heading("Unrecognized parts (out of distribution)")
        rep.line(f"{len(unrec)} part(s) excluded from root-cause grouping:")
        for m in unrec[:30]:
            rep.line(f"   - {m['file']}  (nearest-match {float(m['conf']) * 100:.1f}%)", size=9)

    rep.heading("Factory analytics (all inspections)")
    fp = min(h["factory_pct"], 99.7)
    rep.line(f"Factory health: {fp:.0f}%  (tier: {h['tier']})", bold=True, color="#0b3d1e")
    rep.line(f"Parts inspected: {h['n']}  |  Defect rate: "
             f"{min(h['defect_rate'], 99.7):.0f}%  |  Novel: "
             f"{min(h['novel_rate'], 99.7):.0f}%")
    extra = []
    if dna_p:
        extra.append((dna_p, "Defect-DNA"))
    if cal_p:
        extra.append((cal_p, "Confidence calibration"))
    if extra:
        rep.images(extra, max_w=3.4, max_h=3.0)

    rep.close()
    shutil.rmtree(td, ignore_errors=True)
    return out_path
