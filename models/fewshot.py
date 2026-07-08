"""STAGE 4b — Few-Shot Classifier: Prototypical Network (rare/unseen defects).

Real algorithm:
  - Maintain per-class prototypes = mean of support-set embeddings.
  - classify(embedding): find nearest prototype; flag novel if dist > epsilon.
  - add_support(label, embedding): update prototype incrementally.

Contract:
  FewShotClassifier.classify(embedding, vit_conf, threshold)
    -> (defect_type, confidence, is_novel, distances)
  FewShotClassifier.add_support(label, embedding) -> None
  FewShotClassifier.seed_from_ ViT (cold-start) -> None
"""
from __future__ import annotations

import numpy as np


class FewShotClassifier:
    def __init__(self, novel_threshold: float = 0.3):
        self.prototypes: dict[str, np.ndarray] = {}
        self.support_set: dict[str, list[np.ndarray]] = {}
        self.novel_threshold = novel_threshold

    def add_support(self, label: str, embedding: np.ndarray) -> None:
        self.support_set.setdefault(label, []).append(embedding)
        self.prototypes[label] = self._normalize(
            np.mean(self.support_set[label], axis=0)
        )

    def classify(self, embedding: np.ndarray, vit_conf: float,
                 vit_threshold: float = 0.7):
        # ---- SCAFFOLD DUMMY (replace with real prototype matching) ----
        return "rare_scratch", 0.75, False, {"rare_scratch": 0.25}

    def seed_from_classifier(self, labeled_embeddings: dict) -> None:
        # TODO: batch add from high-confidence ViT predictions
        pass

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
