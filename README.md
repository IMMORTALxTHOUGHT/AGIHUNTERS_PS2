# ForgeMind — Autonomous Visual Manufacturing Intelligence System
### AGIHUNTERS_PS2 · Hackathon team repo

This is the CODE repo. The big **datasets**, **model weights** (GGUF,
`.pt`), and the **virtual env** live ONLY on the SSH box (`/DATA/AGIHUNTERS_PS2`)
and are git-ignored. Git is the source of truth; the box is the runner.

---

## 1. How the 7–10 person team works together

**Principle: git is the source of truth, NOT the SSH box.**
The box is just a GPU + dataset + demo host. Nobody edits `main` on the box.

### One-time (lead)
1. Create the team repo on GitHub/GitLab.
2. This scaffold is already committed on `main`.
3. `git remote add origin <repo-url>` and `git push -u origin main`.

### Every member, every task
```bash
git clone <repo-url>            # on YOUR laptop
git checkout -b feature/<module>
# ... edit your module (against the contracts in ARCHITECTURE.md) ...
git add -A
git commit -m "implement <module>"
git push -u origin feature/<module>
# open a Pull Request -> lead reviews & merges to main
```
On the BOX (for training / running / demo):
```bash
cd /DATA/AGIHUNTERS_PS2
git pull                        # get latest main
source hackathon/bin/activate
python -m pipeline.run --image test.jpg
```

### GPU discipline (24 GB is shared!)
- Designate **1–2 "training owners"** who run PatchCore memory build +
  ViT fine-tune on the box. Everyone else codes CPU-only modules locally.
- Before training: `nvidia-smi` to confirm free VRAM.
- Use a lock file while training: `touch /DATA/AGIHUNTERS_PS2/.gpu.lock`
  and `rm` it after. Coordinate in your team chat.

### Why this avoids chaos
- Modules are **separate files with fixed contracts** (see
  `ARCHITECTURE.md` Part 2). Person A (PatchCore) only has to return
  `(score, heatmap, roi, crop)` of the right types; Person E (debate)
  consumes those types. They never need each other's code to develop.
- The only shared files are `pipeline/run.py`, `requirements.txt`, and
  `README.md` — assign one owner each, edit sequentially.

---

## 2. Task division (up to 10; collapse to 7 by merging lighter ones)

| # | Owner | Modules | Needs GPU? |
|---|-------|---------|-----------|
| A | data + detect | `data/loaders.py`, `models/patchcore.py` | yes (memory build) |
| B | classify | `models/vit_classifier.py`, `models/embedder.py` | yes (fine-tune) |
| C | memory | `data/metadata.py`, `storage/faiss_store.py` | no |
| D | graph | `storage/knowledge_graph.py` | no |
| E | agents | `agents/debate.py`, `agents/moderator.py` | no (local LLM) |
| F | analytics | `analytics/dna_pca.py`, `analytics/calibration.py`, `analytics/health.py` | no |
| G | self-learn | extend `faiss_store` + `knowledge_graph` + sqlite (Stage 12) | no |
| H | dashboard | `dashboard/app.py` | no |
| I | orchestrator | `pipeline/run.py` + integration + testing | no |
| J | docs | `ARCHITECTURE.md` upkeep + final report | no |

---

## 3. Local dev (no GPU needed)

Every module ships with a correctly-shaped **scaffold dummy**, so the whole
pipeline runs end-to-end on a laptop to prove the wiring:
```bash
pip install -r requirements.txt
python -m pipeline.run --image <any-image.jpg>
```
Replace each dummy body with the real implementation on the box.

---

## 4. Box setup (training owners only)
```bash
cd /DATA/AGIHUNTERS_PS2
python -m venv hackathon && source hackathon/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
# start local LLM once:
ollama create qwythos -f Modelfile_qwythos && ollama serve
```

See `ARCHITECTURE.md` for the full stage-by-stage technical deep dive.
