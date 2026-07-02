"""The anti-alignment map (C1): project a model's activations onto d_refuse and classify
the geometry per layer as anti-aligned / neutral / aligned.

Projection is the cosine <y_hat, d_hat> (matches the thesis's ~ -0.15 headline scale). The
SIGN is the load-bearing result: uncensored fine-tunes show <y, d_refuse> < 0 (anti-aligned),
the property naive steering work assumes away. Classification uses the 95% CI over prompts,
so the label is statistically grounded, not a point estimate.

Numeric kernels are numpy (GPU-free, tested); layer_sweep is the only torch-touching part.
"""
from __future__ import annotations

import numpy as np


def projections(acts, d, normalize_y: bool = True) -> np.ndarray:
    """Per-row projection of acts [N, d_model] onto direction d. Cosine by default. Pure."""
    acts = np.asarray(acts, dtype=float)
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    if normalize_y:
        acts = acts / np.clip(np.linalg.norm(acts, axis=-1, keepdims=True), 1e-12, None)
    return acts @ d


def classify_geometry(values, alpha: float = 0.05) -> dict:
    """Mean projection + 95% CI -> {anti-aligned | neutral | aligned} by CI sign, plus a
    per-layer p-value (H0: mean = 0) so BH-FDR can correct across layers (Item 4)."""
    from ..eval.metrics import mean_ci, mean_pvalue

    vals = [float(v) for v in np.asarray(values).ravel()]
    mean, lo, hi = mean_ci(vals, alpha=alpha)
    if hi < 0:
        label = "anti-aligned"
    elif lo > 0:
        label = "aligned"
    else:
        label = "neutral"
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "p_value": mean_pvalue(vals),
            "label": label, "n": len(vals)}


# ── Item 1: confound-controlled anti-alignment measurement ────────────────────
def _unit(d):
    d = np.asarray(d, dtype=float)
    return d / np.linalg.norm(d)


def centered_projections(acts, d, mu_bg) -> np.ndarray:
    """(h - mu_bg) . d_hat per row. Centering by a neutral-corpus mean mu_bg removes the
    confound of where the activation cloud sits; the anti-alignment claim must hold here."""
    return (np.asarray(acts, dtype=float) - np.asarray(mu_bg, dtype=float)) @ _unit(d)


def random_direction_null(acts, shift, K: int = 1000, seed: int = 0) -> np.ndarray:
    """Null distribution of `shift . g_hat` for K directions g drawn from the empirical activation
    covariance (whitened random-direction null). Efficient sampler g = Xc^T z ~ N(0, (N-1)Sigma),
    so no D x D matrix is formed. A direction is special only if the observed projection sits in
    this null's tail."""
    Xc = np.asarray(acts, dtype=float)
    Xc = Xc - Xc.mean(axis=0)
    n = Xc.shape[0]
    if n < 2:
        return np.full(K, np.nan)
    z = np.random.default_rng(seed).standard_normal((n, K))
    g = Xc.T @ z                                            # [D, K] ~ N(0, (n-1)Sigma)
    norms = np.linalg.norm(g, axis=0)
    g = g / np.where(norms > 0, norms, 1.0)
    return np.asarray(shift, dtype=float) @ g               # [K]


def norm_decomposition(vec, d) -> dict:
    """mu = a d_hat + residual: how much of `vec` lies along d (orientation vs magnitude)."""
    vec = np.asarray(vec, dtype=float)
    dh = _unit(d)
    a = float(vec @ dh)
    vn = float(np.linalg.norm(vec))
    return {"along": a, "residual_norm": float(np.linalg.norm(vec - a * dh)), "norm": vn,
            "fraction_along": (a / vn if vn > 0 else float("nan"))}


def anti_alignment_stats(acts, d, mu_bg, *, alpha: float = 0.05, K: int = 1000, seed: int = 0,
                         d_cross=None) -> dict:
    """Confound-controlled per-layer anti-alignment (Item 1). Reports the centered projection
    mean + CI, a whitened random-direction null (percentile + z), an effect size (Cohen's d), a
    norm decomposition of the class shift, and — if `d_cross` (an aligned base model's direction)
    is given — the cross-model cosine and projection. The label is anti-aligned only when the
    centered mean falls below the null's lower tail, not merely below zero."""
    from ..eval.metrics import mean_ci, mean_pvalue

    acts = np.asarray(acts, dtype=float)
    dh = _unit(d)
    mu_bg = np.asarray(mu_bg, dtype=float)
    proj = (acts - mu_bg) @ dh
    mean, lo, hi = mean_ci(proj, alpha=alpha)
    shift = acts.mean(axis=0) - mu_bg
    null = random_direction_null(acts, shift, K=K, seed=seed)
    lo_null = float(np.nanpercentile(null, 100 * alpha / 2))
    hi_null = float(np.nanpercentile(null, 100 * (1 - alpha / 2)))
    label = "anti-aligned" if mean < lo_null else ("aligned" if mean > hi_null else "neutral")
    nstd = float(np.nanstd(null))
    pstd = float(np.std(proj, ddof=1)) if len(proj) > 1 else float("nan")
    out = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "p_value": mean_pvalue(proj),
           "label": label, "n": int(len(proj)),
           "z_score": ((mean - float(np.nanmean(null))) / nstd if nstd > 0 else float("nan")),
           "null_pct": float(np.mean(null < mean) * 100), "null_lo": lo_null, "null_hi": hi_null,
           "cohens_d": (mean / pstd if pstd and not np.isnan(pstd) else float("nan")),
           "norm": norm_decomposition(shift, dh)}
    if d_cross is not None:
        dc = _unit(d_cross)
        out["cross_model_cos"] = float(dh @ dc)
        out["cross_model_mean"] = float(((acts - mu_bg) @ dc).mean())
    return out


def layer_sweep(model, tok, prompts, d_by_layer, *, batch_size: int = 16) -> dict[int, np.ndarray]:
    """Per-layer projection values of `prompts` onto each layer's d_refuse."""
    from .extract import capture_terminal

    layers = sorted(d_by_layer)
    acts = capture_terminal(model, tok, prompts, layers, assistant=None, batch_size=batch_size)
    return {l: projections(acts[l], d_by_layer[l]) for l in layers}


def anti_alignment_map(model, tok, prompts, d_by_layer, *, mu_bg=None, d_cross=None,
                       K: int = 1000, seed: int = 0, batch_size: int = 16) -> dict[int, dict]:
    """The per-model map, one column of the main figure. With `mu_bg` (a neutral-corpus mean per
    layer) it uses the confound-controlled statistics (Item 1); without it, the legacy cosine
    classification. `d_cross` adds the cross-model comparison per layer."""
    from .extract import capture_terminal

    layers = sorted(d_by_layer)
    acts = capture_terminal(model, tok, prompts, layers, assistant=None, batch_size=batch_size)
    out = {}
    for l in layers:
        if mu_bg is not None:
            out[l] = anti_alignment_stats(acts[l], d_by_layer[l], mu_bg[l], K=K, seed=seed,
                                          d_cross=(d_cross[l] if d_cross else None))
        else:
            out[l] = classify_geometry(projections(acts[l], d_by_layer[l]))
    return out
