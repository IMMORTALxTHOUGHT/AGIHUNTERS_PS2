"""STAGE 6 - FAISS Retrieval of historical cases.

Real algorithm: faiss.IndexFlatL2(768).add(all_embeddings); then
D, I = index.search(query, k); map I -> records stored in SQLite / list.

Contract:
  FaissStore.add(embedding, record)
  FaissStore.search(embedding, k) -> list[record]
"""
from __future__ import annotations
import numpy as np


class FaissStore:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.records: list = []   # parallel to the (future) faiss index
        # TODO: self.index = faiss.IndexFlatL2(dim)

    def add(self, emb: np.ndarray, record: dict) -> None:
        self.records.append(record)
        # TODO: self.index.add(emb.reshape(1, -1).astype("float32"))

    def search(self, emb: np.ndarray, k: int = 5) -> list:
        # ---- SCAFFOLD DUMMY (returns first k stored records) ----
        return self.records[:k]
