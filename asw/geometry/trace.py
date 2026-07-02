"""Causal trace: noise-and-restore (Meng et al.) corroborating the cheaper interventional
layer sweep (C3). Protocol from the thesis: corrupt subject-token embeddings with
sigma = 3 * std(embeddings); for each (layer, token) restore the clean mlp.down_proj output
and measure recovery. AIE = P_restore - P_corrupt.

Item 3 upgrades this from BAND-level to DIRECTION-level: a *directional* restore returns only
the d_refuse-component of the (clean - corrupt) delta,

    h_corrupt + ((h_clean - h_corrupt) . d_hat) d_hat,

and we report AIE_directional / AIE_full per layer, with a random-direction restore as control.
A high ratio is the missing causal link between C1/C2 (the direction) and C4 (steering along it);
a low ratio honestly bounds how one-dimensional the refusal mechanism is.

Numpy helpers are unit-tested here; the torch patch/noise hooks and the driver run on the host.
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


def directional_restore(h_corrupt, h_clean, d) -> np.ndarray:
    """Restore only the d-component of the (clean - corrupt) delta:
    h_corrupt + ((h_clean - h_corrupt) . d_hat) d_hat. Pure numpy reference for the hook."""
    hc = np.asarray(h_corrupt, dtype=float)
    delta = np.asarray(h_clean, dtype=float) - hc
    dh = np.asarray(d, dtype=float)
    dh = dh / np.linalg.norm(dh)
    return hc + (delta @ dh)[..., None] * dh


def random_unit_like(d, seed: int = 0) -> np.ndarray:
    """A random unit vector shaped like `d` (the random-direction restore control)."""
    v = np.random.default_rng(seed).standard_normal(np.asarray(d).shape)
    return v / np.linalg.norm(v)


def aie_ratio(aie_directional, aie_full, eps: float = 1e-9) -> np.ndarray:
    """AIE_directional / AIE_full elementwise (nan where |full| < eps). The fraction of the
    band's causal effect that the single direction d_refuse carries."""
    a = np.asarray(aie_directional, dtype=float)
    b = np.asarray(aie_full, dtype=float)
    return np.where(np.abs(b) > eps, a / np.where(np.abs(b) > eps, b, 1.0), np.nan)


def directional_aie_summary(aie_full, aie_directional, aie_random=None) -> dict:
    """Per-layer directional/full AIE ratio + peaks (aie_* are per-layer, mean over positions).
    Includes the random-direction control ratio when provided."""
    full = np.asarray(aie_full, dtype=float)
    ratio = aie_ratio(aie_directional, full)
    out = {"ratio_by_layer": ratio.tolist(), "mean_ratio": float(np.nanmean(ratio)),
           "peak_layer_full": int(np.argmax(full)),
           "peak_layer_directional": int(np.argmax(np.asarray(aie_directional, dtype=float)))}
    if aie_random is not None:
        rnd = aie_ratio(aie_random, full)
        out["random_ratio_by_layer"] = rnd.tolist()
        out["mean_random_ratio"] = float(np.nanmean(rnd))
    return out


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
    """Restore stored clean activations at `positions` of a layer's `site` output.

    mode="full" (default) overwrites the whole vector (band-level trace). mode="directional"
    restores only the component of the (clean - corrupt) delta along `direction` — pass d_refuse
    for the directional AIE, or a random unit vector (see random_unit_like) for the control.
    """

    def __init__(self, model, layer: int, positions, clean, site: str = PATCH_SITE,
                 mode: str = "full", direction=None):
        self.model = model
        self.layer = layer
        self.positions = list(positions)
        self.clean = clean
        self.site = site
        self.mode = mode
        self.direction = direction
        self._h = None

    def __enter__(self):
        import torch

        from ..models.hooks import get_module, hidden_of, repack

        def hook(_m, _i, out):
            h = hidden_of(out)
            c = self.clean.to(h.device, h.dtype)
            if self.mode == "full":
                for p in self.positions:
                    h[:, p, :] = c[:, p, :]
            else:  # directional: restore only the delta component along `direction`
                v = torch.as_tensor(self.direction, device=h.device, dtype=h.dtype)
                v = v / v.norm()
                for p in self.positions:
                    delta = c[:, p, :] - h[:, p, :]
                    comp = (delta * v).sum(dim=-1, keepdim=True)
                    h[:, p, :] = h[:, p, :] + comp * v
            return repack(out, h)

        self._h = get_module(self.model, self.layer, self.site).register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._h:
            self._h.remove()
        return False
