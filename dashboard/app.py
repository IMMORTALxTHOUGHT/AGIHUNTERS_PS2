"""ForgeMind dashboard - working version wired to the real pipeline.

Shows: anomaly heatmap, classification badge, similar past cases (FAISS),
and factory metadata. Uses pipeline.inference.infer_one().

Run: python3 -m dashboard.app
"""
from __future__ import annotations

import html
import os

import cv2
import numpy as np

from config import DASHBOARD_PORT, OUTPUTS_DIR
from pipeline.inference import infer_one
from storage import database
from storage.memory import Memory
from agents.debate import run_debate
from agents.moderator import moderate
from analytics.dna_pca import render_dna_figure
from analytics.health import health
from analytics.calibration import calibration_summary, render_calibration_figure

_memory = Memory()
_last_case_id: dict = {}
_DNA_PATH = str(OUTPUTS_DIR / "dna.png")
_CAL_PATH = str(OUTPUTS_DIR / "calibration.png")

CSS = """
:root, .gradio-container, .dark {
  --body-background-fill: #0d141b;
  --background-fill-primary: #111820;
  --block-background-fill: #111820;
  --block-border-color: #212b36;
  --block-radius: 14px;
  --body-text-color: #e6edf4;
  --body-text-color-subdued: #93a1b0;
  --color-accent: #22c55e;
  --button-primary-background-fill: linear-gradient(92deg,#22c55e,#4ade80);
  --button-primary-text-color: #06140b;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.fm-hero { padding: 28px 4px 8px; }
.fm-hero h1 { margin: 0; font-size: 38px; font-weight: 800; color: #e6edf4; }
.fm-hero h1 .g { color: #22c55e; }
.fm-hero p { color: #93a1b0; font-size: 15px; max-width: 60ch; }
.fm-sec { margin: 24px 4px 4px; font-size: 19px; font-weight: 700; color: #e6edf4; }
.fm-badge { padding: 12px 16px; border-radius: 12px; font-size: 15px; font-weight: 600;
  font-family: ui-monospace, monospace; }
.fm-badge.ok { background: rgba(34,197,94,.10); border: 1px solid rgba(34,197,94,.4); color: #22c55e; }
.fm-badge.novel { background: rgba(255,106,51,.10); border: 1px solid rgba(255,106,51,.4); color: #ff6a33; }
.fm-badge.idle { background: #0d141b; border: 1px solid #212b36; color: #5f6b78; }
.fm-title { font-family: ui-monospace, monospace; font-size: 11.5px; letter-spacing: .1em;
  text-transform: uppercase; color: #93a1b0; margin: 0 0 8px; }
.fm-tbl { width: 100%; border-collapse: collapse; font-size: 13px; }
.fm-tbl th { text-align: left; font-family: ui-monospace, monospace; font-size: 11px;
  text-transform: uppercase; color: #5f6b78; padding: 6px 8px; border-bottom: 1px solid #212b36; }
.fm-tbl td { padding: 7px 8px; border-bottom: 1px solid #212b36; color: #e6edf4; }
.fm-tbl td.sim { font-family: ui-monospace, monospace; color: #3b82f6; }
.fm-empty { color: #5f6b78; font-size: 13px; padding: 12px; }
footer, .footer, .show-api { display: none !important; }
"""

_HERO = """
<div class="fm-hero">
  <h1>Forge<span class="g">Mind</span></h1>
  <p>Upload a part &rarr; <b>what's wrong, where, and similar past cases</b> &mdash;
     explainable and fully local.</p>
</div>
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _badge(result: dict) -> str:
    if result["is_novel_defect"]:
        return ('<div class="fm-badge novel">NOVEL DEFECT &mdash; low confidence, '
                'review needed</div>')
    return (f'<div class="fm-badge ok">&#10004; {_esc(result["defect"])} '
            f'<span style="font-weight:400;color:#93a1b0"> &middot; conf '
            f'{result["vit_confidence"]:.2f}</span></div>')


def _similar_html(result: dict) -> str:
    rows = result.get("similar_cases", [])[:5]
    if not rows:
        return '<div class="fm-empty">No similar cases yet.</div>'
    trs = "".join(
        f'<tr><td>{_esc(r.get("label"))}</td>'
        f'<td class="sim">{r.get("similarity", 0):.3f}</td></tr>'
        for r in rows)
    return (f'<table class="fm-tbl"><thead><tr><th>defect</th>'
            f'<th>similarity</th></tr></thead><tbody>{trs}</tbody></table>')


def _meta_html(result: dict) -> str:
    meta = result.get("metadata", {})
    if not meta:
        return '<div class="fm-empty">No metadata.</div>'
    trs = "".join(
        f'<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>'
        for k, v in meta.items())
    return (f'<table class="fm-tbl"><thead><tr><th>field</th>'
            f'<th>value</th></tr></thead><tbody>{trs}</tbody></table>')


def _kg_html(result: dict) -> str:
    defect = result.get("defect", "unknown")
    know = _memory.get_knowledge(defect)
    summ = know["summary"]

    causes = know["causes"]
    if causes:
        cause_rows = "".join(
            f'<tr><td>{_esc(c["condition"])}</td>'
            f'<td>{c["count"]}</td>'
            f'<td>{c["share"] * 100:.0f}%</td></tr>'
            for c in causes)
        causes_html = (f'<table class="fm-tbl"><thead><tr>'
                        f'<th>associated condition</th><th>seen</th>'
                        f'<th>share</th></tr></thead><tbody>{cause_rows}'
                        f'</tbody></table>')
    else:
        causes_html = '<div class="fm-empty">Not enough data yet — inspect more parts.</div>'

    fix_html = (f'<div class="fm-badge ok" style="white-space:normal">'
                f'&#128161; {_esc(know["fix"])}</div>')

    stats = (f'Recorded {summ["inspections"]} inspections across '
             f'{summ["distinct_defects"]} defect types.')

    return (f'<div class="fm-title">associated conditions for '
            f'{_esc(defect)}</div>{causes_html}'
            f'<div class="fm-title" style="margin-top:14px">recommended fix</div>'
            f'{fix_html}'
            f'<div class="fm-empty" style="margin-top:10px">{stats}</div>')


def _rca_html(image_path) -> str:
    if not image_path:
        return '<div class="fm-empty">Run an inspection first.</div>'
    result = infer_one(image_path)
    defect = result.get("defect", "unknown")
    kg_info = _memory.get_knowledge(defect)
    votes = run_debate(defect, result.get("metadata", {}),
                       result.get("similar_cases", []), kg_info)
    verdict = moderate(votes, defect=defect,
                       metadata=result.get("metadata", {}), kg_info=kg_info)

    vote_rows = "".join(
        f'<tr><td>{_esc(v["role"])}</td>'
        f'<td>{_esc(v["cause"])}</td>'
        f'<td class="sim">{v["conf"]:.2f}</td></tr>'
        for v in votes)
    votes_html = (f'<table class="fm-tbl"><thead><tr><th>specialist</th>'
                  f'<th>hypothesis</th><th>conf</th></tr></thead>'
                  f'<tbody>{vote_rows}</tbody></table>')

    actions = verdict.get("actions") or []
    actions_html = "".join(f'<li>{_esc(a)}</li>' for a in actions) or "<li>—</li>"

    # persist verdict into the case row + grow the knowledge graph
    case_id = _last_case_id.get(image_path)
    _memory.record_rca(case_id, defect, verdict)

    rca_block = (
        f'<div class="fm-title">multi-agent debate (Process / Materials / Reliability)</div>'
        f'{votes_html}'
        f'<div class="fm-title" style="margin-top:14px">winning root cause</div>'
        f'<div class="fm-badge novel" style="white-space:normal">'
        f'{_esc(verdict["winning_cause"])} '
        f'<span style="font-weight:400"> &middot; conf {verdict.get("conf", 0):.2f}</span></div>'
        f'<div class="fm-empty" style="margin-top:8px">{_esc(verdict.get("rationale", ""))}</div>'
        f'<div class="fm-title" style="margin-top:14px">recommended actions</div>'
        f'<ul class="fm-tbl" style="color:#e6edf4;padding-left:18px">'
        f'{actions_html}</ul>'
        f'<div class="fm-empty" style="margin-top:8px">&#10003; Saved to case #{case_id} '
        f'&amp; added to knowledge graph.</div>'
    )
    # refresh the KG panel so the RCA-derived cause/fix edges are visible
    result = infer_one(image_path)
    kg_block = _kg_html(result)
    return (rca_block, kg_block)


def _analytics_payload():
    conn = database.get_connection()
    cases = database.get_cases(conn)
    conn.close()
    if not cases:
        empty = '<div class="fm-empty">No inspections recorded yet — run a few inspections first.</div>'
        return (None, empty, None, empty)

    h = health(cases)
    calib = calibration_summary(cases)
    dna_path = render_dna_figure(cases, _DNA_PATH)
    cal_path = render_calibration_figure(cases, _CAL_PATH)

    tier_cls = ("ok" if h["tier"] == "Good"
                else "novel" if h["tier"] == "Critical" else "idle")
    mach = "".join(f'<tr><td>{_esc(k)}</td><td>{v:.0f}%</td></tr>'
                   for k, v in sorted(h["by_machine"].items()))
    shift = "".join(f'<tr><td>{_esc(k)}</td><td>{v:.0f}%</td></tr>'
                    for k, v in sorted(h["by_shift"].items()))
    health_html = (
        f'<div class="fm-badge {tier_cls}">Factory health: {h["factory_pct"]:.0f}% '
        f'&middot; {h["tier"]}</div>'
        f'<div class="fm-empty" style="margin-top:8px">'
        f'{h["n"]} inspections &middot; defect rate {h["defect_rate"]:.0f}% '
        f'&middot; novel rate {h["novel_rate"]:.0f}%</div>'
        f'<div class="fm-title" style="margin-top:12px">health by machine</div>'
        f'<table class="fm-tbl"><tbody>{mach}</tbody></table>'
        f'<div class="fm-title" style="margin-top:12px">health by shift</div>'
        f'<table class="fm-tbl"><tbody>{shift}</tbody></table>'
    )

    crows = "".join(
        f'<tr><td>{b["range"]}</td><td>{b["count"]}</td>'
        f'<td class="sim">{b["novel_rate"] * 100:.0f}%</td></tr>'
        for b in calib["bins"])
    calib_html = (
        f'<div class="fm-empty">mean conf {calib["mean_conf"]:.2f} &middot; '
        f'overall novel rate {calib["novel_rate"] * 100:.0f}% '
        f'(low-conf bins should be mostly novel)</div>'
        f'<table class="fm-tbl"><thead><tr><th>conf bin</th><th>cases</th>'
        f'<th>% novel</th></tr></thead><tbody>{crows}</tbody></table>'
    )

    return (dna_path, health_html, cal_path, calib_html)


def analyze(image_path):
    if not image_path:
        idle = '<div class="fm-badge idle">Awaiting a part&hellip;</div>'
        empty = '<div class="fm-empty">Awaiting inspection&hellip;</div>'
        return (None, idle, empty, empty, empty)

    result = infer_one(image_path)
    overlay_bgr = result["heatmap_overlay"]
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # self-learning memory: persist every inspection (keep first case id)
    case_id = _memory.record_inspection(result, image_path)
    if case_id is not None:
        _last_case_id[image_path] = case_id

    return (overlay_rgb, _badge(result), _similar_html(result),
            _meta_html(result), _kg_html(result))


def teach(image_path, label):
    if not image_path or not label:
        return '<div class="fm-empty">Load a part and enter the correct label.</div>'
    # re-run inference to get the verified embedding + metadata
    result = infer_one(image_path)
    case_id = _memory.teach(image_path, label, result)
    know = _memory.get_knowledge(label.strip())
    return (f'<div class="fm-badge ok">&#10003; Learned label '
            f'&ldquo;{_esc(label.strip())}&rdquo; (case #{case_id}). '
            f'Fix: {_esc(know["fix"])}</div>'
            f'<div class="fm-empty" style="margin-top:8px">'
            f'{know["summary"]["inspections"]} inspections recorded so far.</div>')


def build():
    import gradio as gr

    with gr.Blocks(title="ForgeMind") as demo:
        gr.HTML(_HERO)

        gr.HTML('<div class="fm-sec">01 &mdash; Inspect a part</div>')
        with gr.Row():
            with gr.Column(scale=5, elem_classes="fm-card"):
                img = gr.Image(label="Part image", type="filepath", height=280)
                run_btn = gr.Button("Inspect", variant="primary")
            with gr.Column(scale=7, elem_classes="fm-card"):
                overlay = gr.Image(label="Anomaly heatmap overlay", height=280)
                badge = gr.HTML('<div class="fm-badge idle">Awaiting a part&hellip;</div>')

        gr.HTML('<div class="fm-sec">02 &mdash; Similar past cases</div>')
        similar_html = gr.HTML('<div class="fm-empty">Awaiting inspection&hellip;</div>')

        gr.HTML('<div class="fm-sec">03 &mdash; Factory metadata</div>')
        meta_html = gr.HTML('<div class="fm-empty">Awaiting inspection&hellip;</div>')

        gr.HTML('<div class="fm-sec">04 &mdash; Knowledge graph &amp; memory</div>')
        kg_html = gr.HTML('<div class="fm-empty">Awaiting inspection&hellip;</div>')

        gr.HTML('<div class="fm-sec">05 &mdash; AI root-cause analysis</div>')
        with gr.Row():
            rca_btn = gr.Button("Explain root cause (multi-agent)", variant="primary")
        rca_html = gr.HTML('<div class="fm-empty">Run an inspection, then ask ForgeMind to explain why.</div>')

        gr.HTML('<div class="fm-sec">06 &mdash; Teach &amp; learn</div>')
        with gr.Row():
            with gr.Column(elem_classes="fm-card"):
                label_in = gr.Textbox(label="Correct defect label",
                                       placeholder="e.g. hairline_crack")
                teach_btn = gr.Button("Teach ForgeMind", variant="primary")
            with gr.Column(elem_classes="fm-card"):
                learn_html = gr.HTML('<div class="fm-empty">Awaiting feedback.</div>')

        gr.HTML('<div class="fm-sec">07 &mdash; Analytics (from inspection history)</div>')
        with gr.Row():
            refresh_btn = gr.Button("Refresh analytics", variant="primary")
        with gr.Row():
            with gr.Column(scale=5, elem_classes="fm-card"):
                dna_img = gr.Image(label="Defect-DNA (PCA)", height=300)
            with gr.Column(scale=5, elem_classes="fm-card"):
                health_html = gr.HTML('<div class="fm-empty">Awaiting data.</div>')
        with gr.Row():
            with gr.Column(scale=6, elem_classes="fm-card"):
                cal_img = gr.Image(label="Confidence calibration", height=260)
            with gr.Column(scale=4, elem_classes="fm-card"):
                calib_html = gr.HTML('<div class="fm-empty">Awaiting data.</div>')

        outputs = [overlay, badge, similar_html, meta_html, kg_html]
        run_btn.click(analyze, inputs=img, outputs=outputs)
        img.upload(analyze, inputs=img, outputs=outputs)
        teach_btn.click(teach, inputs=[img, label_in], outputs=learn_html)
        rca_btn.click(_rca_html, inputs=img, outputs=[rca_html, kg_html])
        refresh_btn.click(_analytics_payload,
                          inputs=None,
                          outputs=[dna_img, health_html, cal_img, calib_html])

    return demo


def launch():
    import gradio as gr
    import os
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build().launch(server_port=DASHBOARD_PORT, css=CSS, share=share)


if __name__ == "__main__":
    launch()
