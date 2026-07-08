"""Reusable Gradio components — assemble the dashboard from these.

Each function returns a gr.* widget ready to be placed in a Blocks layout.
Scaffold — replace placeholder text with real pipeline wiring on the box.
"""
from __future__ import annotations


def image_upload():
    import gradio as gr
    return gr.Image(label="Upload Image", type="filepath")


def heatmap_display():
    import gradio as gr
    return gr.Image(label="Anomaly Heatmap")


def roi_display():
    import gradio as gr
    return gr.Image(label="ROI Crop")


def gallery_display():
    import gradio as gr
    return gr.Gallery(label="Similar Cases (Top 5)")


def kg_graph():
    import gradio as gr
    return gr.HTML(label="Failure Chain Graph")


def dna_scatter():
    import gradio as gr
    return gr.Plot(label="Defect DNA (PCA 2D)")


def calibration_bars():
    import gradio as gr
    return gr.BarPlot(label="Confidence Breakdown",
                      x="component", y="score")


def health_gauges():
    import gradio as gr
    return gr.Dataframe(label="Factory Health")


def fewshot_flag():
    import gradio as gr
    return gr.Textbox(label="Few-Shot / Novel Defect", lines=1)


def rca_output():
    import gradio as gr
    return gr.Textbox(label="Root Cause + Recommendation", lines=10)


def report_download():
    import gradio as gr
    return gr.DownloadButton(label="Download Report")
