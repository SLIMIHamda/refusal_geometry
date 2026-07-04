"""Hook tests run only where torch loads (the GPU host / Kaggle). Skipped on the authoring
box where torch can't load its DLLs (WinError 193)."""
import pytest

try:
    import torch
    from torch import nn
except Exception:  # ImportError or OSError (broken DLLs)
    pytest.skip("torch unavailable on this host", allow_module_level=True)

from asw.models.hooks import Ablator, ActivationCapture, Steerer
from asw.wrapper.steer import WrapperSteer


class ToyBlock(nn.Module):
    def forward(self, x):
        return (x,)  # identity, tuple output like a decoder layer


class ToyModel(nn.Module):
    def __init__(self, n_layers=3, d=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


def test_capture_terminal_token():
    m = ToyModel()
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    with ActivationCapture(m, [0, 1], token_index=-1) as cap:
        m(x)
    a = cap.stacked(0)
    assert a.shape == (2, 4)
    assert torch.allclose(a, x[:, -1, :])


def test_steerer_adds_vector_to_residual():
    m = ToyModel()
    x = torch.zeros(2, 3, 4)
    v = torch.tensor([1.0, 0, 0, 0])
    base = m(x)
    with Steerer(m, {2: v}, alpha=2.0):
        out = m(x)
    assert torch.allclose(out - base, torch.tensor([2.0, 0, 0, 0]).expand_as(out))


def test_steerer_row_mask_protects_benign_rows():
    m = ToyModel()
    x = torch.zeros(2, 3, 4)
    v = torch.tensor([1.0, 0, 0, 0])
    with Steerer(m, {2: v}, alpha=1.0, mask=torch.tensor([True, False])):
        out = m(x)
    assert torch.allclose(out[0], torch.tensor([1.0, 0, 0, 0]).expand(3, 4))  # steered
    assert torch.allclose(out[1], torch.zeros(3, 4))                          # untouched


def test_steerer_accepts_numpy_vector():
    import numpy as np

    m = ToyModel()
    x = torch.zeros(1, 2, 4)
    with Steerer(m, {2: np.array([0.0, 3.0, 0, 0], dtype="float32")}, alpha=1.0):
        out = m(x)
    assert torch.allclose(out, torch.tensor([0.0, 3.0, 0, 0]).expand(1, 2, 4))


def test_ablator_removes_direction_component():
    m = ToyModel()
    x = torch.zeros(1, 2, 4)
    x[..., 0] = 3.0   # component along d
    x[..., 1] = 5.0   # orthogonal component
    with Ablator(m, {2: torch.tensor([2.0, 0, 0, 0])}):   # unit-normalised inside the hook
        out = m(x)
    assert torch.allclose(out[..., 0], torch.zeros(1, 2))          # d-component projected out
    assert torch.allclose(out[..., 1], torch.full((1, 2), 5.0))    # orthogonal untouched


def test_wrappersteer_skip_branch_is_noop():
    # Item 5: a 'skip' neutral layer leaves the residual untouched; a raw_add layer still steers.
    m = ToyModel()
    x = torch.zeros(1, 2, 4)
    v = torch.tensor([1.0, 0, 0, 0])
    with WrapperSteer(m, {1: v, 2: v}, {1: "skip", 2: "raw_add"}, alpha=3.0):
        out = m(x)
    assert torch.allclose(out, torch.tensor([3.0, 0, 0, 0]).expand(1, 2, 4))  # only layer 2 added


def test_ablator_row_mask_protects_rows():
    m = ToyModel()
    x = torch.zeros(2, 2, 4)
    x[..., 0] = 4.0
    with Ablator(m, {2: torch.tensor([1.0, 0, 0, 0])}, mask=torch.tensor([True, False])):
        out = m(x)
    assert torch.allclose(out[0, :, 0], torch.zeros(2))            # ablated row
    assert torch.allclose(out[1, :, 0], torch.full((2,), 4.0))     # untouched row
