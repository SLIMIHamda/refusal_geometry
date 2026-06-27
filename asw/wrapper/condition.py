"""Harmful-input detector in activation space (C4 step 1) — CAST-style condition vector.

A difference-in-means direction between harmful and benign prompt activations at a condition
layer, with a threshold at the midpoint of the two class-projection means. Firing the
intervention only when this detector flags harmful protects XSTest/utility (the conditioning
that prior naive steering lacks). Pure numpy; fit offline and cache.
"""
from __future__ import annotations

import numpy as np


class ConditionVector:
    def __init__(self, direction: np.ndarray, threshold: float):
        self.direction = np.asarray(direction, dtype=float)
        self.threshold = float(threshold)

    @classmethod
    def fit(cls, harmful, benign) -> "ConditionVector":
        harmful = np.asarray(harmful, dtype=float)
        benign = np.asarray(benign, dtype=float)
        d = harmful.mean(axis=0) - benign.mean(axis=0)
        dh = d / np.linalg.norm(d)
        thr = (harmful @ dh).mean() * 0.5 + (benign @ dh).mean() * 0.5
        return cls(dh, thr)

    def score(self, acts) -> np.ndarray:
        return np.asarray(acts, dtype=float) @ self.direction

    def predict(self, acts) -> np.ndarray:
        """True where the input is flagged harmful (projection above threshold)."""
        return self.score(acts) > self.threshold

    def save(self, path) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, direction=self.direction, threshold=np.array(self.threshold))

    @classmethod
    def load(cls, path) -> "ConditionVector":
        data = np.load(path)
        return cls(data["direction"], float(data["threshold"]))
