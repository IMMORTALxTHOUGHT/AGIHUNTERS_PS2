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

from config import DASHBOARD_PORT
from pipeline.inference import infer_one
from storage import database
from storage.memory import Memory

_memory = Memory()

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


def analyze(image_path):
    if not image_path:
        idle = '<div class="fm-badge idle">Awaiting a part&hellip;</div>'
        empty = '<div class="fm-empty">Awaiting inspection&hellip;</div>'
        return (None, idle, empty, empty, empty)

    result = infer_one(image_path)
    overlay_bgr = result["heatmap_overlay"]
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # self-learning memory: persist every inspection
    _memory.record_inspection(result, image_path)

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

        gr.HTML('<div class="fm-sec">05 &mdash; Teach &amp; learn</div>')
        with gr.Row():
            with gr.Column(elem_classes="fm-card"):
                label_in = gr.Textbox(label="Correct defect label",
                                       placeholder="e.g. hairline_crack")
                teach_btn = gr.Button("Teach ForgeMind", variant="primary")
            with gr.Column(elem_classes="fm-card"):
                learn_html = gr.HTML('<div class="fm-empty">Awaiting feedback.</div>')

        outputs = [overlay, badge, similar_html, meta_html, kg_html]
        run_btn.click(analyze, inputs=img, outputs=outputs)
        img.upload(analyze, inputs=img, outputs=outputs)
        teach_btn.click(teach, inputs=[img, label_in], outputs=learn_html)

    return demo


def launch():
    import gradio as gr
    import os
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build().launch(server_port=DASHBOARD_PORT, css=CSS, share=share)


if __name__ == "__main__":
    launch()
