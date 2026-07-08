"""STAGE 12 — Self-Learning Memory + Human Feedback Loop (SPINE).

On *every* inspection it persists the case to SQLite and enriches the
Knowledge Graph. On *Teach* (human feedback) it ingests the corrected label
into FAISS (so future retrievals include the human-verified example) and
persists the store to disk, so the system's memory survives restarts.

This turns ForgeMind from a stateless classifier into a system that
accumulates experience over time.

Contract:
    Memory.record_inspection(result, image_path) -> case_id | None
    Memory.teach(image_path, correct_label, result) -> case_id
    Memory.get_penalties(rca_id) -> list
    Memory.get_knowledge(defect) -> {causes, fix, summary}
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from config import MODEL_WEIGHTS, SQLITE_PATH
from storage import database
from storage.knowledge_graph import KnowledgeGraph


def _active_store():
    """Reach the live FAISS store held by the inference module singletons."""
    try:
        from pipeline import inference
        return inference._store
    except Exception:
        return None


class Memory:
    def __init__(self, db_path: str | Path = SQLITE_PATH):
        self.db_path = Path(db_path)
        database.init_db(database.get_connection(self.db_path))
        self.kg = KnowledgeGraph()
        self._seen_paths: set[str] = set()

    # ---- automatic logging on every inspection ----
    def record_inspection(self, result: dict, image_path: str,
                          skip_dup: bool = True):
        p = str(image_path)
        if skip_dup and p in self._seen_paths:
            return None
        self._seen_paths.add(p)

        conn = database.get_connection(self.db_path)
        case_id = database.insert_case(
            conn,
            defect_type=result.get("defect", "unknown"),
            metadata=result.get("metadata", {}),
            rca={},
            anomaly_score=float(result.get("anomaly_score", 0.0)),
            is_novel=bool(result.get("is_novel_defect", False)),
        )
        conn.close()

        # knowledge-graph enrichment
        self.kg.add_inspection(result.get("defect", "unknown"),
                               result.get("metadata", {}))

        # NOTE: embeddings are NOT pushed into FAISS here — only human-verified
        # Teach entries enter the retrieval index, so it stays high-quality.
        return case_id

    # ---- human feedback ----
    def teach(self, image_path: str, correct_label: str, result: dict | None = None):
        result = result or {}
        label = correct_label.strip()
        conn = database.get_connection(self.db_path)
        case_id = database.insert_case(
            conn,
            defect_type=label,
            metadata=result.get("metadata", {}),
            rca={},
            anomaly_score=float(result.get("anomaly_score", 0.0)),
            is_novel=False,
        )
        conn.close()

        self.kg.add_inspection(label, result.get("metadata", {}))

        # ingest the human-verified embedding into FAISS under the CORRECT
        # label, and persist so the correction survives restarts.
        store = _active_store()
        if store is not None and result.get("embedding") is not None:
            store.add(
                np.asarray(result["embedding"], dtype=np.float32),
                {"path": str(image_path), "label": label, "taught": True},
            )
            try:
                store.save(str(MODEL_WEIGHTS / "faiss_store"))
            except Exception:
                pass
        return case_id

    def get_penalties(self, rca_id: str) -> list:
        conn = database.get_connection(self.db_path)
        rows = conn.execute(
            "SELECT * FROM penalties WHERE rca_id = ?", (rca_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_knowledge(self, defect: str) -> dict:
        return {
            "causes": self.kg.get_causes(defect),
            "fix": self.kg.get_fix(defect),
            "summary": self.kg.summary(),
        }

    def knowledge_graph(self) -> KnowledgeGraph:
        return self.kg
