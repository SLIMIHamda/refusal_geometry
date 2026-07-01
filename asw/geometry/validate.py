"""Construct-validity checks for d_refuse (Review B, Item 2 — blocking).

Four checks, all reusing the extraction/capture spine. They answer: is d_refuse *the* refusal
direction, or an artifact of the extraction template / a confounded difference-in-means?

  ablation   : project d_refuse out of the residual stream at the band layers; refusal on the
               harmful split should collapse. This is the NECESSARY condition — it is meaningful
               on an *aligned* model (which refuses in the first place). Pre-register the drop
               threshold in configs (`validation.ablation_drop_threshold`, default 0.40).
  template   : re-extract with >=5 paraphrased refusal prefixes; report the min pairwise cosine
               per layer. A direction that swings with wording captures form, not function.
  natural    : cos(d_forced, d_natural), where d_natural is a difference-in-means between the
               prompts the model *spontaneously* refuses vs. complies with (native activations
               over a mixed harmful+benign set). Bounds the "refusal-tokens-in-context" confound
               that the forced-prefix protocol could otherwise introduce.
  vs_naive   : cos(d_behavioral, d_naive-DIM) — quantifies what the confound control changed.

Numpy kernels live in extract.py and are unit-tested; this module is the torch orchestration
(host-run), returning a plain, JSON-serialisable metrics dict for the run manifest.
"""
from __future__ import annotations

from .extract import (
    REFUSAL_PREFIXES, capture_terminal, cosine, mean_difference_direction, min_pairwise_cosine,
    naive_dim_direction,
)


def _generate(model, tok, prompts, decoding, *, ablate=None):
    """Greedy generation (T=0), optionally with d_refuse projected out at the band layers."""
    from ..harness.generate import HFGenerator

    gen = HFGenerator(model, tok)
    temp, mnt = decoding["temperatures"][0], decoding["max_new_tokens"]
    if ablate is None:
        return gen.generate(prompts, temperature=temp, max_new_tokens=mnt, seed=0)
    from ..models.hooks import Ablator

    with Ablator(model, ablate):
        return gen.generate(prompts, temperature=temp, max_new_tokens=mnt, seed=0)


def _refusal_rate(labels):
    from ..eval import metrics as M

    k, n = M.counts_from_labels(labels)
    return (k / n if n else float("nan")), k, n


def ablation_test(model, tok, prompts, d_by_layer, judge, decoding, threshold=0.40):
    """Baseline vs d_refuse-ablated refusal rate on the harmful split (necessary condition)."""
    base = judge.label_batch(prompts, _generate(model, tok, prompts, decoding))
    abl = judge.label_batch(prompts, _generate(model, tok, prompts, decoding, ablate=d_by_layer))
    r_base, k_b, n_b = _refusal_rate(base)
    r_abl, k_a, n_a = _refusal_rate(abl)
    drop = r_base - r_abl
    return {"refusal_base": r_base, "refusal_ablated": r_abl, "refusal_drop": drop,
            "threshold": threshold, "passes": bool(drop >= threshold),
            "k_base": k_b, "k_ablated": k_a, "n": n_b}


def template_stability(model, tok, prompts, layers, prefixes=REFUSAL_PREFIXES):
    """Min pairwise cosine per layer across directions extracted with paraphrased prefixes.
    native activations are prefix-independent, so they are captured once."""
    native = capture_terminal(model, tok, prompts, layers, assistant=None)
    dirs = {}
    for pfx in prefixes:
        refusal = capture_terminal(model, tok, prompts, layers, assistant=pfx)
        dirs[pfx] = {l: mean_difference_direction(native[l], refusal[l]) for l in layers}
    return {l: min_pairwise_cosine([dirs[p][l] for p in prefixes]) for l in layers}


def natural_refusal_direction(model, tok, prompts, layers, judge, decoding, min_group=5):
    """d_natural per layer = DIM between the prompts the model spontaneously refuses vs. complies
    with (native activations). Returns (dirs | None, counts)."""
    labels = judge.label_batch(prompts, _generate(model, tok, prompts, decoding))
    refused = [p for p, l in zip(prompts, labels) if l == "refusal"]
    complied = [p for p, l in zip(prompts, labels) if l == "comply"]
    counts = {"n_refused": len(refused), "n_complied": len(complied)}
    if len(refused) < min_group or len(complied) < min_group:
        return None, counts
    a_ref = capture_terminal(model, tok, refused, layers, assistant=None)
    a_com = capture_terminal(model, tok, complied, layers, assistant=None)
    return {l: mean_difference_direction(a_com[l], a_ref[l]) for l in layers}, counts


def behavioral_vs_naive(model, tok, harmful, harmless, layers, d_forced):
    """cos(d_behavioral, d_naive-DIM) per layer, over the same harmful/harmless prompt sets."""
    ah = capture_terminal(model, tok, harmful, layers, assistant=None)
    al = capture_terminal(model, tok, harmless, layers, assistant=None)
    d_naive = {l: naive_dim_direction(ah[l], al[l]) for l in layers}
    return {l: cosine(d_forced[l], d_naive[l]) for l in layers}


def run_validation(model, tok, d_forced, *, harmful, harmless, layers, judge, decoding,
                   threshold=0.40):
    """Run all four construct-validity checks; return a JSON-serialisable metrics dict."""
    layers = list(layers)
    d_band = {l: d_forced[l] for l in layers}

    abl = ablation_test(model, tok, harmful, d_band, judge, decoding, threshold)
    templ = template_stability(model, tok, harmful, layers)
    d_nat, nat_counts = natural_refusal_direction(model, tok, harmful + harmless, layers,
                                                  judge, decoding)
    natural = {l: cosine(d_forced[l], d_nat[l]) for l in layers} if d_nat else None
    vs_naive = behavioral_vs_naive(model, tok, harmful, harmless, layers, d_forced)

    return {
        "ablation": abl,
        "template_stability_cos": {str(l): v for l, v in templ.items()},
        "template_stability_min": float(min(templ.values())),
        "natural_refusal_cos": ({str(l): v for l, v in natural.items()} if natural else None),
        "natural_refusal_counts": nat_counts,
        "behavioral_vs_naive_cos": {str(l): v for l, v in vs_naive.items()},
        "layers": layers,
    }
