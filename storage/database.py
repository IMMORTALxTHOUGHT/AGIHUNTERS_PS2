"""SQLite database layer — low-level CRUD for cases and feedback.

Higher-level orchestration lives in storage/memory.py.
"""
from __future__ import annotations

import sqlite3
import json
from pathlib import Path

from config import SQLITE_PATH


def get_connection(db_path: str | Path = SQLITE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create tables if they don't exist."""
    close = conn is None
    conn = conn or get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            defect_type TEXT NOT NULL,
            metadata_json TEXT,
            rca_json TEXT,
            anomaly_score REAL,
            is_novel INTEGER DEFAULT 0,
            vit_confidence REAL,
            embedding_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS penalties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rca_id TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    if close:
        conn.close()


def insert_case(conn: sqlite3.Connection, defect_type: str, metadata: dict,
                rca: dict, anomaly_score: float, is_novel: bool,
                vit_confidence: float | None = None,
                embedding_id: int | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO cases (defect_type, metadata_json, rca_json, anomaly_score, is_novel, vit_confidence, embedding_id) "
        "VALUES (?,?,?,?,?,?,?)",
        (defect_type, json.dumps(metadata), json.dumps(rca),
         anomaly_score, int(is_novel), vit_confidence, embedding_id),
    )
    conn.commit()
    return cur.lastrowid


def get_cases(conn: sqlite3.Connection, limit: int = 100000) -> list:
    rows = conn.execute(
        "SELECT defect_type, metadata_json, rca_json, anomaly_score, "
        "is_novel, vit_confidence, created_at FROM cases ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "defect_type": r["defect_type"],
            "metadata": json.loads(r["metadata_json"] or "{}"),
            "rca": json.loads(r["rca_json"] or "{}"),
            "anomaly_score": r["anomaly_score"],
            "is_novel": bool(r["is_novel"]),
            "vit_confidence": r["vit_confidence"],
            "created_at": r["created_at"],
        })
    return out


def insert_penalty(conn: sqlite3.Connection, rca_id: str) -> None:
    conn.execute("INSERT INTO penalties (rca_id) VALUES (?)", (rca_id,))
    conn.commit()
