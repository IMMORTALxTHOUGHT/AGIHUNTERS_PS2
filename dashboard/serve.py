"""ForgeMind HUD — a thin, dependency-free web server.

Serves the standalone "Forge HUD" frontend (dashboard/static/index.html) and
wraps the real pipeline so the custom UI drives the same engine as the Gradio
dashboard:

    pipeline.inference.infer_one  -> PatchCore + ViT + FAISS
    agents.debate / agents.moderator -> multi-agent RCA
    storage.memory / knowledge_graph -> self-learning memory + KG
    dashboard.report.build_pdf -> downloadable full report

Run:  python3 -m dashboard.serve   (or: GRADIO_SHARE=1 python3 -m dashboard.serve)
"""
from __future__ import annotations

import json
import base64
import os
import glob
import uuid
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

from config import DASHBOARD_PORT, OUTPUTS_DIR, REPORTS_DIR
from dashboard import app as engine


def _b64_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _system_status():
    summ = engine._memory.get_knowledge("__any__")["summary"]
    return {
        "inspections": summ.get("inspections", 0),
        "defects": summ.get("distinct_defects", 0),
    }


def run_inspection(image_b64: str, name: str) -> dict:
    # decode + persist the uploaded part
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    ext = ".png"
    if name and name.lower().endswith((".jpg", ".jpeg")):
        ext = ".jpg"
    path = str(OUTPUTS_DIR / f"hud_{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(raw)

    result = engine.infer_one(path)

    # self-learning memory: persist the inspection (first case id wins)
    case_id = engine._memory.record_inspection(result, path)
    if case_id is not None:
        engine._last_case_id[path] = case_id

    # original + heatmap overlay for the viewport
    orig = cv2.imread(path)
    overlay = result.get("heatmap_overlay")
    orig_b64 = _b64_png(orig) if orig is not None else ""
    overlay_b64 = _b64_png(overlay) if overlay is not None else ""

    # analysis panels (reuse the exact HTML builders) — everything EXCEPT the
    # slow multi-agent RCA, which is fetched separately so the fast panels
    # render immediately while the agents debate in the background.
    report_html, dna_path, cal_path = engine._analytics_payload()

    dna_b64 = _b64_png(cv2.imread(dna_path)) if dna_path and os.path.exists(dna_path) else ""
    cal_b64 = _b64_png(cv2.imread(cal_path)) if cal_path and os.path.exists(cal_path) else ""

    return {
        "label": result.get("defect", "unknown"),
        "confidence": float(result.get("vit_confidence", 0.0)),
        "anomaly_score": float(result.get("anomaly_score", 0.0)),
        "badge_html": engine._badge(result),
        "severity_html": engine._severity_badge(result),
        "viewport": overlay_b64,
        "orig": orig_b64,
        "similar_html": engine._similar_html(result),
        "meta_html": engine._meta_html(result),
        "kg_html": engine._kg_html(result),
        "analytics_html": report_html,
        "dna": dna_b64,
        "calibration": cal_b64,
        "system": _system_status(),
        "path": path,
    }


def run_rca(path: str) -> dict:
    """The multi-agent root-cause debate — kept separate so the rest of the
    inspection renders instantly while this (LLM-bound) step runs."""
    rca_block, kg_block = engine._rca_html(path)
    return {"rca_html": rca_block, "kg_html": kg_block}


def sample_part() -> dict:
    """Pick a random image from the on-box datasets so the UI has a
    one-click demo part without the user hunting for a file."""
    roots = ["datasets/mvtec", "datasets/neu/NEU-DET", "datasets/dagm"]
    import random
    found = []
    for r in roots:
        p = os.path.join(os.getcwd(), r)
        if os.path.isdir(p):
            for dp, _, fs in os.walk(p):
                for fn in fs:
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                        found.append(os.path.join(dp, fn))
    if not found:
        return {"error": "No sample images found under datasets/ on this box."}
    pick = random.choice(found)
    img = cv2.imread(pick)
    return {"image": _b64_png(img), "name": os.path.basename(pick)}


def run_batch(folder: str, images: list) -> dict:
    """Batch inspection: a folder path on the box and/or multiple uploaded
    images. Each part is classified + heatmapped and auto-logged so Analytics
    and the Knowledge Graph grow from the batch."""
    paths: list = []
    if folder:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            paths += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    for im in (images or []):
        b64 = im.get("image", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        ext = ".jpg" if (im.get("name", "")).lower().endswith((".jpg", ".jpeg")) else ".png"
        p = str(OUTPUTS_DIR / f"batch_{uuid.uuid4().hex}{ext}")
        with open(p, "wb") as f:
            f.write(raw)
        paths.append(p)
    paths = sorted(set(paths))
    if not paths:
        return {"count": 0, "rows": [], "gallery": []}

    rows, gallery = [], []
    for p in paths:
        try:
            result = engine.infer_one(p)
        except Exception:
            continue
        engine._memory.record_inspection(result, p)  # grow memory + KG
        defect = result.get("defect", "unknown")
        sev = {"high": "High", "mid": "Mid", "low": "Low",
               "review": "Review"}[engine._severity(result)]
        rows.append({
            "file": os.path.basename(p),
            "defect": defect,
            "conf": engine._pct(result.get("vit_confidence", 0.0)),
            "severity": sev,
            "anomaly": f'{result.get("anomaly_score", 0.0):.2f}',
            "novel": "YES" if result.get("is_novel_defect") else "no",
        })
        if len(gallery) < 24 and result.get("heatmap_overlay") is not None:
            gallery.append(_b64_png(result["heatmap_overlay"]))
    return {"count": len(rows), "rows": rows, "gallery": gallery}


def build_report(path: str) -> bytes:
    result = engine.infer_one(path)
    defect = result.get("defect", "unknown")
    kg_info = engine._memory.get_knowledge(defect)
    votes = engine.run_debate(
        defect, result.get("metadata", {}),
        result.get("similar_cases", []), kg_info)
    verdict = engine.moderate(
        votes, defect=defect,
        metadata=result.get("metadata", {}), kg_info=kg_info)
    pdf_path = engine.build_pdf(path, result, verdict, str(REPORTS_DIR))
    with open(pdf_path, "rb") as f:
        return f.read()


_STATIC = os.path.join(os.path.dirname(__file__), "static", "index.html")


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, *a):  # quiet
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path == "/api/sample":
            try:
                body = json.dumps(sample_part()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, str(e))
            return
        if urlparse(self.path).path in ("/", "/index.html"):
            try:
                with open(_STATIC, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.end_headers()
                self.wfile.write(body)
            except FileNotFoundError:
                self.send_error(404, "Frontend not found")
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self.send_error(400)
            return

        if path == "/api/inspect":
            try:
                out = run_inspection(data.get("image", ""), data.get("name", ""))
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            except Exception as e:  # surface backend errors to the UI
                traceback.print_exc()
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/rca":
            try:
                out = run_rca(data.get("path", ""))
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            except Exception as e:
                traceback.print_exc()
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/batch":
            try:
                out = run_batch(data.get("folder", ""), data.get("images", []))
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            except Exception as e:
                traceback.print_exc()
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/report":
            try:
                pdf = build_report(data.get("path", ""))
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition",
                                 'attachment; filename="forge_report.pdf"')
                self.send_header("Content-Length", str(len(pdf)))
                self._cors()
                self.end_headers()
                self.wfile.write(pdf)
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)


def launch():
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    port = DASHBOARD_PORT
    if share:
        # expose via a public tunnel when requested (same flag as the Gradio app)
        try:
            import pyngrok.conf as _
            from pyngrok import ngrok
            public = ngrok.connect(port)
            print(f"Forge HUD public URL: {public.public_url}")
        except Exception:
            print("(GRADIO_SHARE=1 set but pyngrok unavailable — serving locally)")
    print(f"Forge HUD  ->  http://localhost:{port}")
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    launch()
