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


# ── detector characterization (Review B, item 4) ──────────────────────────────
# The wrapper's whole claim is that conditioning protects benign / over-refusal inputs, so the
# detector must be reported on HELD-OUT data, not train accuracy — especially its firing rate on
# XSTest (benign-but-scary prompts): a high FPR there means the wrapper over-refuses.
def roc_auc(scores_pos, scores_neg) -> float:
    """AUC that the detector ranks a harmful (pos) above a benign (neg) prompt. Rank-based
    (Mann-Whitney), tie-aware. nan if either class is empty."""
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    n1, n2 = pos.size, neg.size
    if n1 == 0 or n2 == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def tpr_fpr(scores_pos, scores_neg, threshold: float) -> tuple[float, float]:
    """(TPR on harmful, FPR on benign) at a firing threshold (detector fires when score > tau)."""
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    tpr = float((pos > threshold).mean()) if pos.size else float("nan")
    fpr = float((neg > threshold).mean()) if neg.size else float("nan")
    return tpr, fpr


def threshold_sweep(scores_harm, scores_benign, scores_over, taus) -> list[dict]:
    """Firing rates vs threshold: TPR (harmful), FPR (benign), and FPR on the over-refusal set
    (XSTest). This is the sensitivity curve — how detection and over-refusal move as tau shifts."""
    sh = np.asarray(scores_harm, dtype=float)
    sb = np.asarray(scores_benign, dtype=float)
    so = np.asarray(scores_over, dtype=float)
    return [{"tau": float(t),
             "tpr": float((sh > t).mean()) if sh.size else float("nan"),
             "fpr_benign": float((sb > t).mean()) if sb.size else float("nan"),
             "fpr_over": float((so > t).mean()) if so.size else float("nan")} for t in taus]
