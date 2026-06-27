"""Forward-hook machinery: capture residual-stream activations and add steering vectors.

This is the shared spine for extraction (WS2), the anti-alignment projection (C1), the
causal trace (WS3), the wrapper (C4), and the attacks (C5). Per the thesis constraint we
never edit weights in place under device_map="auto"; hooks are registered after dispatch
and every vector is moved to the hidden state's device/dtype inside the hook (sharding
splits layers across GPUs).

Hook sites on a Llama/Qwen/Mistral block `model.model.layers[i]`:
  "block"          -> decoder-layer output (residual stream; default add point, Arditi-style)
  "mlp"            -> MLP submodule output
  "mlp.down_proj"  -> down-projection output (the trace patch site, WS3)
  "attn"           -> self-attention submodule output (attention trace arm)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch


def get_module(model, layer_idx: int, site: str = "block"):
    block = model.model.layers[layer_idx]
    if site == "block":
        return block
    if site == "mlp":
        return block.mlp
    if site == "mlp.down_proj":
        return block.mlp.down_proj
    if site == "attn":
        return block.self_attn
    raise ValueError(f"unknown hook site '{site}'")


def hidden_of(out):
    """The hidden-state tensor from a module output (tuple-aware)."""
    return out[0] if isinstance(out, tuple) else out


def repack(out, h):
    """Put a (possibly modified) hidden state back into the module's output shape."""
    return (h, *out[1:]) if isinstance(out, tuple) else h


class ActivationCapture:
    """Capture residual-stream activations at `layers` during forward passes.

    `token_index=-1` keeps the terminal-token activation (the thesis's terminal-token DIM);
    pass `token_index=None` to keep the full [batch, seq, d] tensor (needed by the trace).
    Captures accumulate across forward calls; `stacked(layer)` concatenates them.
    """

    def __init__(self, model, layers: Iterable[int], site: str = "block", token_index: int | None = -1):
        self.model = model
        self.layers = list(layers)
        self.site = site
        self.token_index = token_index
        self._buf: dict[int, list[torch.Tensor]] = {l: [] for l in self.layers}
        self._handles: list = []

    def _make(self, layer: int):
        def hook(_m, _i, out):
            h = hidden_of(out).detach()
            if self.token_index is not None:
                h = h[:, self.token_index, :]
            self._buf[layer].append(h.to("cpu", torch.float32))
        return hook

    def __enter__(self):
        for l in self.layers:
            self._handles.append(get_module(self.model, l, self.site).register_forward_hook(self._make(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def stacked(self, layer: int) -> torch.Tensor:
        return torch.cat(self._buf[layer], dim=0)

    def clear(self):
        for l in self.layers:
            self._buf[l].clear()


class Steerer:
    """Add `alpha * vectors[layer]` to the residual stream at each layer (raw addition).

    `vectors` maps layer -> 1D direction. An optional per-row boolean `mask` (length = batch)
    restricts steering to selected rows — this is how the conditional wrapper applies the
    intervention to harmful inputs only while leaving benign rows untouched in the same batch.
    """

    def __init__(self, model, vectors: dict[int, torch.Tensor], alpha: float = 1.0,
                 site: str = "block", mask: torch.Tensor | None = None):
        self.model = model
        self.vectors = vectors
        self.alpha = alpha
        self.site = site
        self.mask = mask
        self._handles: list = []

    def _make(self, vec):
        def hook(_m, _i, out):
            h = hidden_of(out)
            v = torch.as_tensor(vec, device=h.device, dtype=h.dtype)  # numpy or tensor
            if self.mask is None:
                h = h + self.alpha * v
            else:
                m = self.mask.to(h.device).view(-1, *([1] * (h.dim() - 1))).to(h.dtype)
                h = h + self.alpha * v * m
            return repack(out, h)
        return hook

    def __enter__(self):
        for l, vec in self.vectors.items():
            self._handles.append(get_module(self.model, l, self.site).register_forward_hook(self._make(vec)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


@contextmanager
def no_grad_eval(model):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        yield model
    if was_training:
        model.train()
