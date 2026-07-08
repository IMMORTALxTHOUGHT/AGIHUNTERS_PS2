"""Thin OpenAI-compatible client for the LOCAL LLM (Ollama / Qwythos).

Uses plain `requests` so there is no hard dependency on the openai SDK.
The endpoint is OpenAI-compatible: {LLM_BASE_URL}/chat/completions.

All calls degrade gracefully: if the model is unreachable, chat() returns
None and callers fall back to a deterministic (KG-grounded) hypothesis.
"""
from __future__ import annotations

import json
import re

from config import LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def available() -> bool:
    """True if the local Ollama server is up and lists our model."""
    if not _HAS_REQUESTS:
        return False
    try:
        base = LLM_BASE_URL.rsplit("/v1", 1)[0]
        r = requests.get(base + "/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        models = [m.get("name", "") for m in r.json().get("models", [])]
        return any(LLM_MODEL.split(":")[0] in m for m in models)
    except Exception:
        return False


def chat(system: str, user: str, temperature: float | None = None,
         timeout: int = 90) -> str | None:
    if not _HAS_REQUESTS:
        return None
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def parse_json(text: str | None, default=None):
    if not text:
        return default if default is not None else {}
    m = _JSON_RE.search(text)
    if not m:
        return default if default is not None else {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return default if default is not None else {}
