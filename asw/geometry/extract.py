"""Behavioral-contrast extraction of the refusal direction d_refuse (C2).

Same prompt, two forced continuations, terminal-token difference-in-means:
  refusal    : the prompt with a refusal sentence begun in the assistant turn
  compliance : the same prompt with an ACCEDING sentence begun in the assistant turn
  d_refuse[layer] = normalize( mean(refusal_acts) - mean(compliance_acts) )

Pairing on the *same* prompts cancels the topic/length confounds a naive harmful-vs-harmless
difference-in-means leaves in (the C2 confound-control claim).

WHY BOTH POLES ARE CONTINUATIONS (the 2026-07-24 fix). The first estimator contrasted the
refusal continuation against the bare generation point (`assistant=None`). Those two states end
on *different tokens* — the "." of a refusal sentence versus the `<|im_start|>assistant\\n`
header — so the dominant component of the difference was "the assistant is mid-sentence", not
"the assistant is refusing". Measured on dolphin-2.9-llama3-8b, that direction was 0.599 cosine
with a placebo continuation differing by a SINGLE token ("I cannot help" -> "I can help") and
0.52 with an explicit compliance continuation: a majority-format vector. Steering with it could
not induce refusal at any strength; it only destroyed the model once alpha approached ||h||.
Contrasting refusal against compliance puts both poles mid-sentence at a matched position, so
the format component cancels and what survives is stance. `format_direction` measures the
residue directly, and `validation.format_confound_max` gates it.

RESIDUAL RISK, named rather than assumed away: pairs that differ only by negation risk encoding
the token "not"/"cannot" instead of the refusal function. The pair list therefore varies the
discourse framing as well, and the causal claim still rests on the ablation test in validate.py
(project d_refuse out of an ALIGNED model and refusal must collapse), not on this construction.

Numeric kernels operate on numpy arrays (testable without a GPU); capture_terminal is the
only torch-touching part (lazy import) and runs on the GPU host. d_refuse is returned as a
numpy unit vector per layer — it serializes trivially and Steerer accepts it directly.
"""
from __future__ import annotations

import numpy as np

REFUSAL_PREFIX = "I cannot help with that request."
COMPLIANCE_PREFIX = "I can help with that request."

# Paired continuations: same surface frame, opposite stance. Pairing (rather than drawing two
# independent lists) is what holds length and register matched pole-to-pole, so the difference
# isolates stance. The first three are near-minimal pairs — the tightest possible confound
# control; the last two vary the discourse marker so the direction cannot be purely the negation
# token. Extraction pools all pairs; template_stability re-extracts from each pair separately and
# reports the min pairwise cosine, which is what justifies pooling them.
CONTRAST_PREFIX_PAIRS = [
    (REFUSAL_PREFIX,                                COMPLIANCE_PREFIX),
    ("I can't assist with that.",                   "I can assist with that."),
    ("I will not provide that information.",        "I will provide that information."),
    ("Unfortunately, I cannot provide that.",       "Certainly, I can provide that."),
    ("I'm sorry, but I won't do that.",             "Of course, I will do that."),
]

# Refusal poles alone — kept because callers and older caches refer to them by this name.
REFUSAL_PREFIXES = [r for r, _ in CONTRAST_PREFIX_PAIRS]
COMPLIANCE_PREFIXES = [c for _, c in CONTRAST_PREFIX_PAIRS]


def mean_difference_direction(native, refusal) -> np.ndarray:
    """Unit refusal direction from paired activation banks [N, d]. Pure (numpy)."""
    native = np.asarray(native, dtype=float)
    refusal = np.asarray(refusal, dtype=float)
    d = refusal.mean(axis=0) - native.mean(axis=0)
    return d / np.linalg.norm(d)


def naive_dim_direction(harmful, harmless) -> np.ndarray:
    """Naive harmful-vs-harmless difference-in-means (the confounded estimator C2 improves on):
    unit(mean(harmful) - mean(harmless)). Reported against d_behavioral to quantify what the
    confound control actually changed. Pure (numpy)."""
    return mean_difference_direction(harmless, harmful)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def min_pairwise_cosine(directions) -> float:
    """Minimum cosine over all pairs of unit directions — the template-stability lower bound.
    Returns nan for fewer than two directions. Pure (numpy)."""
    ds = [np.asarray(d, dtype=float) for d in directions]
    pairs = [cosine(ds[i], ds[j]) for i in range(len(ds)) for j in range(i + 1, len(ds))]
    return float(min(pairs)) if pairs else float("nan")


def _format(tok, prompt: str, assistant: str | None) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return text + assistant if assistant else text


def capture_terminal(model, tok, prompts, layers, *, assistant=None, site="block",
                     batch_size=16) -> dict[int, np.ndarray]:
    """Terminal-token residual activations per layer for `prompts` -> {layer: [N, d]} numpy.

    `assistant` may be None, a single string (broadcast to every prompt), or a per-prompt list
    (used to teacher-force each prompt's own natural refusal continuation, Item 2)."""
    from ..models.hooks import ActivationCapture, no_grad_eval

    layers = list(layers)
    assistants = assistant if isinstance(assistant, (list, tuple)) else [assistant] * len(prompts)
    with ActivationCapture(model, layers, site=site, token_index=-1) as cap, no_grad_eval(model):
        for i in range(0, len(prompts), batch_size):
            texts = [_format(tok, p, a)
                     for p, a in zip(prompts[i:i + batch_size], assistants[i:i + batch_size])]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            model(**enc)
    return {l: cap.stacked(l).numpy() for l in layers}


def extract_drefuse(model, tok, prompts, layers, *, pairs=None, site="block",
                    batch_size=16) -> dict[int, np.ndarray]:
    """Extract d_refuse per layer: refusal continuations vs COMPLIANCE continuations.

    Pools every pair in `pairs` (default CONTRAST_PREFIX_PAIRS) into one difference-in-means, so
    the direction is not hostage to a single surface form. Costs 2 forward passes per prompt per
    pair — pass a one-pair list for a fast run."""
    pairs = list(pairs if pairs is not None else CONTRAST_PREFIX_PAIRS)
    refusal_acts: dict[int, list] = {l: [] for l in layers}
    comply_acts: dict[int, list] = {l: [] for l in layers}
    for refusal_pfx, comply_pfx in pairs:
        r = capture_terminal(model, tok, prompts, layers, assistant=refusal_pfx,
                             site=site, batch_size=batch_size)
        c = capture_terminal(model, tok, prompts, layers, assistant=comply_pfx,
                             site=site, batch_size=batch_size)
        for l in layers:
            refusal_acts[l].append(r[l])
            comply_acts[l].append(c[l])
    return {l: mean_difference_direction(np.concatenate(comply_acts[l]),
                                         np.concatenate(refusal_acts[l])) for l in layers}


def extract_drefuse_vs_native(model, tok, prompts, layers, *, refusal_prefix=REFUSAL_PREFIX,
                              site="block", batch_size=16) -> dict[int, np.ndarray]:
    """The PRE-FIX estimator: refusal continuation vs the bare generation point.

    Retained deliberately. It is the direction whose format confound broke the first steering
    pass, and reporting cos(d_refuse, d_vs_native) quantifies what the fix actually changed — a
    measured delta rather than an asserted one. Not for steering."""
    native = capture_terminal(model, tok, prompts, layers, assistant=None,
                              site=site, batch_size=batch_size)
    refusal = capture_terminal(model, tok, prompts, layers, assistant=refusal_prefix,
                               site=site, batch_size=batch_size)
    return {l: mean_difference_direction(native[l], refusal[l]) for l in layers}


def format_direction(model, tok, prompts, layers, *, compliance_prefix=COMPLIANCE_PREFIX,
                     site="block", batch_size=16) -> dict[int, np.ndarray]:
    """The pure-format axis: unit(mean(compliance_acts) - mean(native_acts)).

    Both poles are non-refusing — they differ only in whether the assistant turn has begun. So
    this direction carries the "mid-sentence" component and nothing about refusal. A valid
    d_refuse must be close to orthogonal to it; the pre-fix estimator scored ~0.6 here, which is
    exactly why it could not steer. Consumed by validate.format_confound."""
    native = capture_terminal(model, tok, prompts, layers, assistant=None,
                              site=site, batch_size=batch_size)
    comply = capture_terminal(model, tok, prompts, layers, assistant=compliance_prefix,
                              site=site, batch_size=batch_size)
    return {l: mean_difference_direction(native[l], comply[l]) for l in layers}


def layer_consistency(d_by_layer: dict[int, np.ndarray]) -> dict[int, float]:
    """Cosine between consecutive layers' directions — sanity check that d_refuse is a
    coherent direction across the steering band rather than per-layer noise."""
    layers = sorted(d_by_layer)
    return {layers[i]: cosine(d_by_layer[layers[i - 1]], d_by_layer[layers[i]])
            for i in range(1, len(layers))}


def save_drefuse(path, d_by_layer: dict[int, np.ndarray], meta: dict | None = None) -> None:
    """Cache d_refuse to an .npz (keys 'L{layer}'), with an optional sidecar JSON of meta."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"L{l}": np.asarray(v) for l, v in d_by_layer.items()})
    if meta is not None:
        Path(str(path) + ".json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")


def load_drefuse(path) -> dict[int, np.ndarray]:
    data = np.load(path)
    return {int(k[1:]): data[k] for k in data.files}
