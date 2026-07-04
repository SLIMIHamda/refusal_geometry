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


# ── prompt-clustered CI (Review B, item 5) ────────────────────────────────────
def cluster_bootstrap_rate_ci(df, *, cluster: str = "prompt_id", value: str = "refused",
                              alpha: float = 0.05, B: int = 2000, seed: int = 0):
    """Prompt-clustered bootstrap CI for a rate. Returns (rate, lo, hi, n_clusters).

    `df` has one row per replicate (seed x temperature) with a `cluster` id and a 0/1 `value`.
    Resamples CLUSTERS (prompts) with replacement B times; each resample's rate is the mean over
    all replicates of the sampled clusters. Responses to the SAME prompt across seeds/temps are
    correlated (at T=0 they are identical), so pooling replicates as independent Bernoulli trials
    inflates n and understates the interval — this is the honest CI. Degrades gracefully: one
    cluster -> lo=hi=rate; it also matches an ordinary bootstrap when there is one row per prompt."""
    if df is None or len(df) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    g = df.groupby(cluster)[value]
    sums = g.sum().to_numpy(dtype=float)
    counts = g.count().to_numpy(dtype=float)
    total = counts.sum()
    if total == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rate = float(sums.sum() / total)
    n_clusters = sums.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(B, n_clusters))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return rate, float(lo), float(hi), int(n_clusters)


# ── pre-registered alpha selection (Review B, item 3) ─────────────────────────
def select_alpha(points):
    """Pick the steering strength alpha that maximises (harmful-refusal − over-refusal) on the
    TUNING split — the pre-registered selection rule (Item 3). Ties resolve to the SMALLEST alpha
    (least intervention). `points` is a list of {alpha, refusal_harmful, refusal_over}. Returns the
    selected alpha, or None if empty. Frozen before the headline HarmBench/XSTest runs, this is what
    closes the tuning-leakage objection."""
    best_obj, best_alpha = None, None
    for p in sorted(points, key=lambda q: q["alpha"]):
        if p.get("refusal_harmful") is None or p.get("refusal_over") is None:
            continue
        obj = p["refusal_harmful"] - p["refusal_over"]
        if best_obj is None or obj > best_obj + 1e-12:      # strict -> ties keep the smaller alpha
            best_obj, best_alpha = obj, p["alpha"]
    return best_alpha


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


# ── multiplicity, power, and the crossover interaction (Review B, Item 4) ──────
def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """Benjamini-Hochberg FDR. Returns (qvalues, reject) aligned to input order.

    With ~30 layers x per-layer CIs, uncorrected labels are guaranteed to contain noise; the
    anti-alignment map should be read on q-values, not raw per-layer significance."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]           # enforce monotonic step-up
    q_full = np.empty(n)
    q_full[order] = np.clip(q, 0.0, 1.0)
    return q_full, q_full <= alpha


def required_n(delta: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> float:
    """Per-group n to detect a `delta`-point change from baseline p0 (two-sided, two-proportion,
    normal approximation)."""
    if delta == 0:
        return float("inf")
    from scipy.stats import norm

    p1 = min(max(p0 + delta, 0.0), 1.0)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p0 + p1) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar))
           + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / (abs(delta) ** 2))


def min_detectable_effect(n: int, p0: float = 0.5, alpha: float = 0.05,
                          power: float = 0.8) -> float:
    """Smallest difference in proportions detectable at sample size `n` (inverts required_n).
    At n=100, p0=0.5 this is ~0.19, which is why sub-20-point claims need n>=300."""
    lo, hi = 0.0, 1.0 - p0
    for _ in range(60):
        mid = (lo + hi) / 2
        if required_n(mid, p0, alpha, power) <= n:
            hi = mid
        else:
            lo = mid
    return hi


def mean_pvalue(values: Sequence[float]) -> float:
    """Two-sided p-value that the mean of `values` differs from 0 (one-sample t-test). Feeds
    the per-layer anti-alignment test so BH-FDR can be applied across layers."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    from scipy.stats import ttest_1samp

    return float(ttest_1samp(arr, 0.0).pvalue)


def crossover_interaction(df, *, outcome: str = "refused", operator: str = "operator",
                          geometry: str = "geometry", group: str = "prompt_id",
                          ref_operator: str = "none", ref_geometry: str = "aligned") -> dict:
    """Population-averaged logistic GEE of `outcome ~ operator * geometry`, clustered on `group`
    (prompt-level dependence; seeds are replicate rows). The interaction terms ARE the paper's
    headline statistic: raw_add should help anti-aligned models while projection-amplification
    should help aligned models yet hurt anti-aligned ones.

    `df` is one row per (prompt, operator, geometry, seed) with a 0/1 `outcome`. Needs
    statsmodels; returns the interaction coefficients with 95% CI and p (host-run)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    formula = (f"{outcome} ~ C({operator}, Treatment(reference='{ref_operator}'))"
               f" * C({geometry}, Treatment(reference='{ref_geometry}'))")
    res = smf.gee(formula, groups=group, data=df, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = res.conf_int()
    inter = {name: {"coef": float(res.params[name]),
                    "ci_lo": float(ci.loc[name, 0]), "ci_hi": float(ci.loc[name, 1]),
                    "p": float(res.pvalues[name])}
             for name in res.params.index if ":" in name}
    return {"interactions": inter, "n_obs": int(res.nobs),
            "n_groups": int(df[group].nunique()),
            "params": {k: float(v) for k, v in res.params.items()}}
