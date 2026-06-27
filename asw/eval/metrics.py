"""Statistics for the harness (Briefing 1, Axis C).

Pure-numpy at import time; scipy/sklearn are lazy-imported only inside the functions
that need them, so the spine still imports on a machine without them.

Key design points motivated by the briefings:
- Clopper-Pearson gives a *meaningful upper bound* when k=0 ("0.0% ASR" is a hypothesis,
  not a result): refusal_rate_ci(0, 100) -> (0.0, 0.0, ~0.036).
- Seed-level aggregation reports mean ± 95% CI (bootstrap) across >=3 seeds.
- paired_test backs claims like "wrapper vs CAST" deltas.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

REFUSAL = "refusal"
COMPLY = "comply"


# ── point estimate from labels ────────────────────────────────────────────────
def counts_from_labels(labels: Iterable[str]) -> tuple[int, int]:
    """Return (k_refusal, n_scored). 'unclear' is excluded from the denominator."""
    scored = [x for x in labels if x in (REFUSAL, COMPLY)]
    return sum(x == REFUSAL for x in scored), len(scored)


def rate(labels: Iterable[str]) -> float:
    k, n = counts_from_labels(labels)
    return float("nan") if n == 0 else k / n


# ── binomial confidence intervals ─────────────────────────────────────────────
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 100*(1-alpha)% interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval — no scipy needed; good default for moderate n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def refusal_rate_ci(
    k: int, n: int, alpha: float = 0.05, method: str = "clopper-pearson"
) -> tuple[float, float, float]:
    """Return (rate, lo, hi). Default exact CI so the k=0 upper bound is meaningful."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    lo, hi = (
        clopper_pearson(k, n, alpha) if method == "clopper-pearson" else wilson_ci(k, n)
    )
    return p, lo, hi


# ── seed-level aggregation ────────────────────────────────────────────────────
def mean_ci(
    values: Sequence[float], alpha: float = 0.05, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean ± bootstrap 95% CI across per-seed values (>=3 seeds expected)."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


# ── paired significance (e.g. wrapper vs CAST on matched prompts/seeds) ────────
def paired_test(a: Sequence[float], b: Sequence[float], kind: str = "wilcoxon") -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired_test needs equal-length matched samples")
    from scipy import stats

    if kind == "wilcoxon":
        res = stats.wilcoxon(a, b)
    elif kind == "t":
        res = stats.ttest_rel(a, b)
    else:
        raise ValueError(kind)
    return {"kind": kind, "statistic": float(res[0]), "p_value": float(res[1]),
            "mean_delta": float((a - b).mean())}


# ── dual-scorer agreement (Axis C: report at M1) ──────────────────────────────
def agreement(labels_a: Sequence[str], labels_b: Sequence[str]) -> dict:
    if len(labels_a) != len(labels_b):
        raise ValueError("agreement needs equal-length label sequences")
    n = len(labels_a)
    raw = sum(x == y for x, y in zip(labels_a, labels_b)) / n if n else float("nan")
    out = {"n": n, "raw_agreement": raw, "cohen_kappa": float("nan")}
    try:
        from sklearn.metrics import cohen_kappa_score

        out["cohen_kappa"] = float(
            cohen_kappa_score(list(labels_a), list(labels_b),
                              labels=[REFUSAL, COMPLY, "unclear"])
        )
    except Exception:
        pass
    return out
