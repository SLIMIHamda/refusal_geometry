"""Geometry branch (C4 step 2) + the steering hook.

Two operators, chosen per the measured sign of <y, d_refuse> (C1):
  raw_add          (anti-aligned): h + alpha * d_hat        — inject refusal regardless of sign
  project_amplify  (aligned/neutral): h + alpha * (h . d_hat) * d_hat  — amplify the model's
                   OWN refusal component; strengthens refusal where geometry is positive and
                   *fails* where it is negative (the symmetric C1 falsification prediction).

Numpy reference ops are unit-tested; WrapperSteer is the torch hook used during generation.
"""
from __future__ import annotations

import numpy as np


def branch_for_label(label: str, neutral_op: str = "project") -> str:
    """Pick the operator from a geometry label (from projection.classify_geometry). Anti-aligned
    gets raw_add, aligned gets project-amplify; NEUTRAL layers use `neutral_op` — swept in the
    3-way micro-ablation (project vs raw_add vs skip, Item 5) to justify their treatment with data
    rather than assertion."""
    if label == "anti-aligned":
        return "raw_add"
    if label == "neutral":
        return neutral_op
    return "project"


def _unit(d):
    d = np.asarray(d, dtype=float)
    return d / np.linalg.norm(d)


def op_raw_add(h, d, alpha: float) -> np.ndarray:
    """h + alpha * d_hat (broadcast over leading dims). Pure numpy reference."""
    return np.asarray(h, dtype=float) + alpha * _unit(d)


def op_project_amplify(h, d, alpha: float) -> np.ndarray:
    """h + alpha * (h . d_hat) * d_hat. Pure numpy reference."""
    h = np.asarray(h, dtype=float)
    dh = _unit(d)
    comp = h @ dh
    return h + alpha * comp[..., None] * dh


def op_skip(h, d, alpha: float) -> np.ndarray:
    """Identity: leave the residual stream untouched (the 'skip' branch for neutral layers)."""
    return np.asarray(h, dtype=float)


class WrapperSteer:
    """Apply the per-layer geometry branch to the residual stream during a forward pass.

    `branch_by_layer[l]` in {"raw_add","project"}; `mask` (batch bool) restricts steering to
    rows the condition flagged harmful, leaving benign rows untouched in the same batch.
    """

    def __init__(self, model, d_by_layer, branch_by_layer, alpha: float, mask=None, site="block"):
        self.model = model
        self.d_by_layer = d_by_layer
        self.branch_by_layer = branch_by_layer
        self.alpha = alpha
        self.mask = mask
        self.site = site
        self._handles: list = []

    def _make(self, layer: int):
        import torch

        from ..models.hooks import hidden_of, repack

        d = self.d_by_layer[layer]
        branch = self.branch_by_layer[layer]

        def hook(_m, _i, out):
            h = hidden_of(out)
            if branch == "skip":                 # neutral-layer no-op (Item 5 micro-ablation)
                return out
            dh = torch.as_tensor(d, device=h.device, dtype=h.dtype)
            dh = dh / dh.norm()
            if branch == "raw_add":
                delta = self.alpha * dh
            else:  # project amplify: scale the activation's own component along d
                comp = (h * dh).sum(dim=-1, keepdim=True)
                delta = self.alpha * comp * dh
            if self.mask is not None:
                m = self.mask.to(h.device).view(-1, *([1] * (h.dim() - 1))).to(h.dtype)
                delta = delta * m
            return repack(out, h + delta)

        return hook

    def __enter__(self):
        from ..models.hooks import get_module

        for l in self.d_by_layer:
            self._handles.append(
                get_module(self.model, l, self.site).register_forward_hook(self._make(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False
