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


def layer_sweep(model, tok, prompts, d_by_layer, *, batch_size: int = 16) -> dict[int, np.ndarray]:
    """Per-layer projection values of `prompts` onto each layer's d_refuse."""
    from .extract import capture_terminal

    layers = sorted(d_by_layer)
    acts = capture_terminal(model, tok, prompts, layers, assistant=None, batch_size=batch_size)
    return {l: projections(acts[l], d_by_layer[l]) for l in layers}


def anti_alignment_map(model, tok, prompts, d_by_layer, *, batch_size: int = 16) -> dict[int, dict]:
    """The per-model map: {layer: classify_geometry(...)} — one column of the main figure."""
    sweep = layer_sweep(model, tok, prompts, d_by_layer, batch_size=batch_size)
    return {l: classify_geometry(v) for l, v in sweep.items()}
