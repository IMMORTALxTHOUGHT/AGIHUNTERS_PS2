# ForgeMind — Explainable Industrial Defect Intelligence

ForgeMind is a fully **local, offline** defect-detection system for heavy-industry
part inspection. It doesn't just say *what* is wrong — it shows *where* (anomaly
heatmap), retrieves *similar past cases*, explains *why* (multi-agent LLM
root-cause analysis), and **learns** from every inspection and from human
feedback.

Everything runs on-premise: vision models, vector retrieval, the knowledge
graph, and the reasoning LLM (Ollama) — no cloud, no API keys.

## Features

| Capability | What it does |
|---|---|
| **Anomaly detection** | PatchCore memory-bank + heatmap localizes the defect. |
| **Defect classification** | ViT-B/16 assigns a defect type with confidence. |
| **Similar-case retrieval** | FAISS finds the closest historical examples. |
| **Knowledge graph** | Accumulates defect → cause/fix associations from data **and** LLM analysis. |
| **Multi-agent RCA** | Process / Materials / Reliability agents debate the root cause; a moderator synthesizes a verdict + actions. |
| **Self-learning memory** | Every inspection is logged; human "Teach" corrections enter the retrieval index. |
| **Analytics** | Defect-DNA (PCA), factory-health tiers, confidence calibration. |
| **Batch mode** | Inspect a whole folder at once. |

## Pipeline

```
image
  │
  ├─ PatchCore ─────────────► anomaly score + heatmap overlay
  ├─ ViT classifier ────────► defect type + confidence
  ├─ Embedder (256-d) ──────► fingerprint
  │                              │
  └──────────────────────────────┘
                                 ▼
                            FAISS store ──► similar past cases
                                 │
                 Knowledge Graph + Self-Learning Memory (SQLite)
                                 │
                    Multi-Agent LLM Root-Cause (Ollama)
```

## Quick start

```bash
# 1. (on the GPU box) create env + install
python3 -m venv hackathon && source activate hackathon
pip install -r requirements.txt

# 2. ensure the local LLM is running (for root-cause analysis)
ollama pull qwythos
ollama serve          # serves OpenAI-compatible API at http://localhost:11434/v1

# 3. run the dashboard
python3 -m dashboard.app
# open http://localhost:7860  (or GRADIO_SHARE=1 python3 -m dashboard.app for a public link)
```

> Models/weights and datasets are large and kept on the box only (see
> `.gitignore`). On first run the FAISS store is built automatically from the
> datasets; it is then cached under `models/weights/`.

## Using the dashboard

1. **Inspect a part** — upload an image → heatmap, class badge, similar cases, factory metadata.
2. **Knowledge graph & memory** — associated conditions + recommended fix, accumulated across all inspections.
3. **AI root-cause analysis** — *Explain root cause* launches the 3-agent debate → winning cause, rationale, and actions (saved to the case).
4. **Teach & learn** — type the correct label to record a human-verified example into FAISS.
5. **Analytics** — *Refresh analytics* shows Defect-DNA scatter, factory-health tier, and confidence calibration.
6. **Batch inspect** — enter a folder path *or* select multiple images to process them all at once, auto-logging each one.

## Project layout

```
config.py              central config (paths, thresholds, dims, LLM endpoint)
pipeline/inference.py  infer_one() — single entry point tying the pipeline together
models/
  patchcore.py         anomaly detection + heatmap
  vit_classifier.py    defect classification (train + predict)
  embedder.py          256-d normalized embedding for FAISS
storage/
  faiss_store.py       vector retrieval
  database.py          SQLite cases/feedback
  memory.py            self-learning memory (logs + Teach → FAISS/KG)
  knowledge_graph.py   data-driven causal graph (grows from inspections + RCA)
agents/
  llm.py               OpenAI-compatible client for local Ollama
  prompts.py           specialist + moderator prompts
  debate.py            multi-agent root-cause debate
  moderator.py         synthesizes winning cause + actions
analytics/
  dna_pca.py           Defect-DNA PCA (pure numpy)
  health.py            factory-health tiers
  calibration.py       confidence calibration summary
dashboard/app.py       Gradio UI (8 sections)
scripts/visual_test.py verifies a single image as a 2×3 figure
```

## Notes

- **Factory metadata is synthetic but deterministic** (seeded from the filename)
  so demos are reproducible. Swap `data/metadata.py` for real sensor telemetry
  to productionize.
- The system degrades gracefully: if Ollama is unavailable, root-cause analysis
  falls back to knowledge-graph-grounded hypotheses, and analytics still render.
- Novel/low-confidence defects are flagged and can be taught via the dashboard.
