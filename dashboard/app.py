"""STAGE 14 - Gradio Dashboard + Explainable Report.

Real algorithm: build a Gradio Blocks UI with image + heatmap overlay,
ROI crop, retrieved-cases gallery, KG graph (pyvis), DNA scatter,
calibration bars, health gauges, RCA textbox, and a download button.
Wire img.upload -> pipeline.run.full_pipeline(path).

Contract: build() -> gr.Blocks ; launch() starts the server.
"""
from __future__ import annotations


def build():
    import gradio as gr

    def _placeholder(path):
        return "scaffold - wire pipeline.run.full_pipeline(path) here"

    with gr.Blocks(title="ForgeMind") as demo:
        gr.Markdown("# ForgeMind - Autonomous Manufacturing Intelligence")
        img = gr.Image(label="Upload Image", type="filepath")
        out = gr.Textbox(label="Root Cause + Recommendation", lines=8)
        img.upload(_placeholder, inputs=img, outputs=out)
    return demo


def launch():
    build().launch(server_port=7860)


if __name__ == "__main__":
    launch()
    # TODO: img.upload(fn=full_pipeline, inputs=img, outputs=[...all widgets...])
