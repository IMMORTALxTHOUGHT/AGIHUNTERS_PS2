# ForgeMind — Autonomous Visual Manufacturing Intelligence System

Upload a factory photo → get back: what's wrong, where, why, and what to do about it. All explainable, all local.

---

## Quick start (no GPU)

```bash
pip install -r requirements.txt
python -m pipeline.run --image any-image.jpg
```

Every module ships with scaffold dummies — the pipeline runs end-to-end on a laptop to prove wiring. Replace stubs with real logic on the GPU box.

---

## How it works in 30 seconds

```
Upload → PatchCore detects anomaly → ViT classifies it
        → if ViT is unsure, Few-Shot classifier handles rare defects
        → FAISS finds similar past cases
        → 3 AI agents debate the root cause → moderator decides
        → System remembers everything (learns over time)
        → Dashboard shows the full story
```

The system **learns** — every inspection updates FAISS, the knowledge graph, and the few-shot support set so it gets smarter over time.

---

## Project layout

| Directory | What's in it |
|---|---|
| `data/` | Image loaders + synthetic factory metadata |
| `models/` | PatchCore (detector), ViT (classifier), Few-Shot (rare defects), Embedder |
| `agents/` | Multi-agent debate: 3 specialists + moderator + prompt templates |
| `storage/` | FAISS vector search, knowledge graph, SQLite database, self-learning memory |
| `analytics/` | Defect DNA (PCA), confidence calibration, factory health score |
| `dashboard/` | Gradio UI + reusable components |
| `pipeline/` | Orchestrator (CLI entry), inference, utilities |
| `config.py` | One file with all paths and hyperparameters |

---

## Team task board

Pair names with modules. Each module has a fixed contract (inputs → outputs) so you never need each other's code to develop.

| # | Owner | Modules | Needs GPU? |
|---|---|---|---|
| A | data + detect | `data/loaders.py`, `models/patchcore.py` | yes (memory build) |
| B | classify + embed | `models/vit_classifier.py`, `models/embedder.py` | yes (fine-tune) |
| C | few-shot | `models/fewshot.py` | no |
| D | memory + graph | `storage/faiss_store.py`, `storage/knowledge_graph.py`, `storage/database.py` | no |
| E | memory loop | `storage/memory.py` (self-learning + feedback) | no |
| F | agents | `agents/debate.py`, `agents/moderator.py`, `agents/prompts.py` | no (local LLM) |
| G | analytics | `analytics/dna_pca.py`, `analytics/calibration.py`, `analytics/health.py` | no |
| H | dashboard | `dashboard/app.py`, `dashboard/components.py` | no |
| I | orchestrator | `pipeline/run.py`, `pipeline/inference.py`, `pipeline/utils.py` | no |
| J | docs + config | `config.py`, `ARCHITECTURE.md`, `README.md` | no |

---

## Git workflow

```bash
# every member on their laptop
git checkout -b feature/<module>
# ... edit your module ...
git add -A && git commit -m "your message"
git push -u origin feature/<module>
# open a PR → lead reviews & merges to main

# on the GPU box (for training / demo):
git pull
python -m pipeline.run --image test.jpg
```

**Rule:** git is source of truth, NOT the SSH box. Nobody edits main on the box.

---

## Box setup (training owners only)

```bash
cd /DATA/AGIHUNTERS_PS2
python -m venv hackathon && source hackathon/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
ollama create qwythos -f Modelfile_qwythos && ollama serve
```

See `ARCHITECTURE.md` for the full stage-by-stage technical deep dive.
