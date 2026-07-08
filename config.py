"""Central configuration — paths, hyperparameters, thresholds.

All modules import from here instead of hardcoding magic values.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent

# --- Paths ---
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_WEIGHTS = ROOT / "models" / "weights"
STORAGE_DIR = ROOT / "storage"
SQLITE_PATH = STORAGE_DIR / "forge_mind.db"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"

# --- Model params ---
VIT_INPUT_SIZE = 224
VIT_CONF_THRESHOLD = 0.7
VIT_EMBED_DIM = 768
VIT_PROJ_DIM = 256
PATCHCORE_BACKBONE = "wide_resnet50_2"
PATCHCORE_FEAT_DIM = 1024
PATCHCORE_MEM_BANK_PATH = MODEL_WEIGHTS / "mem_bank.npy"

# --- Few-shot ---
FSL_NOVEL_THRESHOLD = 0.3
FSL_CONFIDENCE_THRESHOLD = 0.7

# --- FAISS ---
FAISS_DIM = 256
FAISS_TOP_K = 5

# --- LLM ---
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwythos"
LLM_TEMPERATURE = 0.2

# --- Dashboard ---
DASHBOARD_PORT = 7860
