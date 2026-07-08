"""Shared utilities for the pipeline.

Pure helper functions — no stage logic, no model imports.
"""
from __future__ import annotations

import json
from pathlib import Path


def read_image_path(path: str) -> str:
    """Validate and normalise an image path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return str(p.resolve())


def serialise_result(result: dict) -> str:
    """JSON-serialise a pipeline result dict (handles numpy types)."""
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return super().default(obj)
    return json.dumps(result, indent=2, cls=_Encoder)


def timestamp() -> str:
    """ISO-8601 timestamp string."""
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
