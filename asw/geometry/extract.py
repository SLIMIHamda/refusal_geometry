"""Behavioral-contrast extraction of the refusal direction d_refuse (C2).

Same prompt, two conditions, terminal-token difference-in-means (the thesis protocol):
  native  : the formatted prompt at the generation point (model about to comply)
  refusal : the same prompt with a forced refusal prefix begun in the assistant turn
  d_refuse[layer] = normalize( mean(refusal_acts) - mean(native_acts) )

Pairing on the *same* prompts cancels topic/length confounds a naive harmful-vs-harmless
difference-in-means leaves in (the C2 confound-control claim).

Numeric kernels operate on numpy arrays (testable without a GPU); capture_terminal is the
only torch-touching part (lazy import) and runs on the GPU host. d_refuse is returned as a
numpy unit vector per layer — it serializes trivially and Steerer accepts it directly.
"""
from __future__ import annotations

import numpy as np

REFUSAL_PREFIX = "I cannot help with that request."


def mean_difference_direction(native, refusal) -> np.ndarray:
    """Unit refusal direction from paired activation banks [N, d]. Pure (numpy)."""
    native = np.asarray(native, dtype=float)
    refusal = np.asarray(refusal, dtype=float)
    d = refusal.mean(axis=0) - native.mean(axis=0)
    return d / np.linalg.norm(d)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _format(tok, prompt: str, assistant: str | None) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return text + assistant if assistant else text


def capture_terminal(model, tok, prompts, layers, *, assistant=None, site="block",
                     batch_size=16) -> dict[int, np.ndarray]:
    """Terminal-token residual activations per layer for `prompts` -> {layer: [N, d]} numpy."""
    from ..models.hooks import ActivationCapture, no_grad_eval

    layers = list(layers)
    with ActivationCapture(model, layers, site=site, token_index=-1) as cap, no_grad_eval(model):
        for i in range(0, len(prompts), batch_size):
            texts = [_format(tok, p, assistant) for p in prompts[i:i + batch_size]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            model(**enc)
    return {l: cap.stacked(l).numpy() for l in layers}


def extract_drefuse(model, tok, prompts, layers, *, refusal_prefix=REFUSAL_PREFIX,
                    site="block", batch_size=16) -> dict[int, np.ndarray]:
    """Extract d_refuse per layer via behavioral-contrast pairing."""
    native = capture_terminal(model, tok, prompts, layers, assistant=None,
                              site=site, batch_size=batch_size)
    refusal = capture_terminal(model, tok, prompts, layers, assistant=refusal_prefix,
                               site=site, batch_size=batch_size)
    return {l: mean_difference_direction(native[l], refusal[l]) for l in layers}


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
