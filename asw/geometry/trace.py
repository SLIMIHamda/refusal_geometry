"""Causal trace: noise-and-restore (Meng et al.) corroborating the cheaper interventional
layer sweep (C3). Protocol from the thesis: corrupt subject-token embeddings with
sigma = 3 * std(embeddings); for each (layer, token) restore the clean mlp.down_proj output
and measure recovery. AIE = P_restore - P_corrupt.

Numpy helpers (sigma, AIE grid) are unit-tested here; the torch patch/noise hooks and the
driver run on the GPU host.
"""
from __future__ import annotations

import numpy as np

PATCH_SITE = "mlp.down_proj"


# ── pure helpers ──────────────────────────────────────────────────────────────
def noise_sigma(embed_weight, k: float = 3.0) -> float:
    """sigma = k * std(embedding matrix). Accepts numpy or torch (via np.asarray)."""
    return float(k) * float(np.asarray(embed_weight, dtype=float).std())


def aie_grid(restore_grid, p_corrupt: float) -> np.ndarray:
    """AIE[layer, pos] = P_restore[layer, pos] - P_corrupt. Pure."""
    return np.asarray(restore_grid, dtype=float) - float(p_corrupt)


def peak_layer(aie_by_layer) -> int:
    """Index of the layer with the largest mean AIE (the refusal-mediation zone)."""
    arr = np.asarray(aie_by_layer, dtype=float)
    return int(np.argmax(arr.mean(axis=1) if arr.ndim == 2 else arr))


# ── torch hooks (host-run) ────────────────────────────────────────────────────
class EmbeddingNoise:
    """Add Gaussian noise of scale `sigma` to input embeddings at `positions`."""

    def __init__(self, model, positions, sigma: float, seed: int = 0):
        self.model = model
        self.positions = list(positions)
        self.sigma = sigma
        self.seed = seed
        self._h = None

    def __enter__(self):
        import torch

        g = torch.Generator(device="cpu").manual_seed(self.seed)

        def hook(_m, _i, out):
            noise = torch.randn(out.shape, generator=g).to(out.device, out.dtype) * self.sigma
            mask = torch.zeros_like(out)
            mask[:, self.positions, :] = 1.0
            return out + noise * mask

        self._h = self.model.get_input_embeddings().register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._h:
            self._h.remove()
        return False


class RestorePatch:
    """Overwrite `site` output at `positions` with stored clean activations (single layer)."""

    def __init__(self, model, layer: int, positions, clean, site: str = PATCH_SITE):
        self.model = model
        self.layer = layer
        self.positions = list(positions)
        self.clean = clean
        self.site = site
        self._h = None

    def __enter__(self):
        from ..models.hooks import get_module, hidden_of, repack

        def hook(_m, _i, out):
            h = hidden_of(out)
            c = self.clean.to(h.device, h.dtype)
            for p in self.positions:
                h[:, p, :] = c[:, p, :]
            return repack(out, h)

        self._h = get_module(self.model, self.layer, self.site).register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._h:
            self._h.remove()
        return False
