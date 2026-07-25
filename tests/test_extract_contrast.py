"""The extraction-contrast fix (2026-07-24), tested against a synthetic reproduction of the defect.

The real failure: `d_refuse` was extracted as (refusal continuation - bare generation point). Those
two states end on different tokens, so the difference was dominated by "the assistant is
mid-sentence" rather than "the assistant is refusing" — measured at 0.599 cosine with a
one-token-flipped placebo on dolphin-2.9-llama3-8b, and unable to steer refusal at any alpha.

Here that geometry is built by hand so the claim is checkable without a GPU:

    native[i]     = base[i]
    compliance[i] = base[i] + 3*FMT              <- the format shift alone
    refusal[i]    = base[i] + 3*FMT + 1*STANCE   <- the same shift, plus stance

with FMT and STANCE orthogonal. The old estimator must recover 3*FMT + STANCE (contaminated);
the new one must recover STANCE alone (clean). That is the whole fix, as arithmetic.
"""
import numpy as np
import pytest

from asw.geometry import extract as E
from asw.geometry.validate import format_confound

D = 8
FMT = np.zeros(D); FMT[0] = 1.0            # "assistant has begun a sentence"
STANCE = np.zeros(D); STANCE[1] = 1.0      # "the assistant is refusing"
LAYERS = [13, 14]
N = 6

# per-prompt baseline activations — arbitrary but fixed, and identical across conditions, which is
# what the same-prompt pairing buys us
BASE = np.arange(N * D, dtype=float).reshape(N, D) * 0.01


def _fake_capture(monkeypatch, *, fmt_scale=3.0, stance_scale=1.0):
    """Patch capture_terminal so `assistant` selects the synthetic condition.

    Patched in BOTH namespaces: validate.py does `from .extract import capture_terminal`, binding
    its own reference, so patching only extract's global leaves validate calling the real (torch-
    backed) function."""
    from asw.geometry import validate as V

    def fake(model, tok, prompts, layers, *, assistant=None, site="block", batch_size=16):
        if assistant is None:
            acts = BASE
        elif assistant in E.COMPLIANCE_PREFIXES:
            acts = BASE + fmt_scale * FMT
        elif assistant in E.REFUSAL_PREFIXES:
            acts = BASE + fmt_scale * FMT + stance_scale * STANCE
        else:
            raise AssertionError(f"unexpected continuation: {assistant!r}")
        return {l: acts.copy() for l in layers}

    monkeypatch.setattr(E, "capture_terminal", fake)
    monkeypatch.setattr(V, "capture_terminal", fake)


def test_new_estimator_recovers_stance_only(monkeypatch):
    _fake_capture(monkeypatch)
    d = E.extract_drefuse(None, None, ["p"] * N, LAYERS)
    for l in LAYERS:
        assert np.allclose(d[l], STANCE, atol=1e-9), f"layer {l} picked up format"


def test_old_estimator_is_contaminated(monkeypatch):
    """Pins the defect: the pre-fix direction is mostly format, and that is why it could not steer."""
    _fake_capture(monkeypatch)
    d_old = E.extract_drefuse_vs_native(None, None, ["p"] * N, LAYERS)
    expected = (3.0 * FMT + STANCE) / np.linalg.norm(3.0 * FMT + STANCE)
    assert np.allclose(d_old[13], expected, atol=1e-9)
    assert E.cosine(d_old[13], FMT) > 0.9          # ~0.949 — dominated by sentence-state


def test_fix_changes_the_direction(monkeypatch):
    """The two estimators must actually disagree, or the fix is cosmetic."""
    _fake_capture(monkeypatch)
    new = E.extract_drefuse(None, None, ["p"] * N, LAYERS)[13]
    old = E.extract_drefuse_vs_native(None, None, ["p"] * N, LAYERS)[13]
    assert E.cosine(new, old) < 0.4


def test_format_direction_isolates_the_confound(monkeypatch):
    _fake_capture(monkeypatch)
    d_fmt = E.format_direction(None, None, ["p"] * N, LAYERS)
    for l in LAYERS:
        assert np.allclose(d_fmt[l], FMT, atol=1e-9)


def test_format_confound_gate_passes_new_and_fails_old(monkeypatch):
    """The gate must discriminate — a check that passes everything is not a check."""
    _fake_capture(monkeypatch)
    prompts = ["p"] * N
    new = E.extract_drefuse(None, None, prompts, LAYERS)
    old = E.extract_drefuse_vs_native(None, None, prompts, LAYERS)

    ok = format_confound(None, None, prompts, LAYERS, new, threshold=0.35)
    bad = format_confound(None, None, prompts, LAYERS, old, threshold=0.35)
    assert ok["passes"] and ok["worst"] < 1e-9
    assert not bad["passes"] and bad["worst"] > 0.9


def test_format_confound_uses_absolute_value(monkeypatch):
    """Contamination of either sign is contamination."""
    _fake_capture(monkeypatch)
    flipped = {l: -E.format_direction(None, None, ["p"] * N, LAYERS)[l] for l in LAYERS}
    res = format_confound(None, None, ["p"] * N, LAYERS, flipped, threshold=0.35)
    assert not res["passes"] and res["worst"] == pytest.approx(1.0)


def test_pooling_over_pairs_averages_surface_forms(monkeypatch):
    """Extraction pools every pair; a single-pair run is a supported fast path and, when the
    stance component is shared, must agree with the pooled direction."""
    _fake_capture(monkeypatch)
    prompts = ["p"] * N
    pooled = E.extract_drefuse(None, None, prompts, LAYERS)[13]
    one_pair = E.extract_drefuse(None, None, prompts, LAYERS,
                                 pairs=[E.CONTRAST_PREFIX_PAIRS[0]])[13]
    assert E.cosine(pooled, one_pair) == pytest.approx(1.0, abs=1e-9)


def test_template_stability_runs_per_pair(monkeypatch):
    """Signature changed from a prefix list to contrast PAIRS; it must still return one min
    pairwise cosine per layer. With a shared stance component every pair agrees -> 1.0."""
    from asw.geometry.validate import template_stability

    _fake_capture(monkeypatch)
    out = template_stability(None, None, ["p"] * N, LAYERS)
    assert set(out) == set(LAYERS)
    for l in LAYERS:
        assert out[l] == pytest.approx(1.0, abs=1e-9)


def test_contrast_pairs_are_well_formed():
    """Every pair must be a genuine stance flip on a matched frame, not a length mismatch."""
    assert len(E.CONTRAST_PREFIX_PAIRS) >= 5
    for refusal, comply in E.CONTRAST_PREFIX_PAIRS:
        assert refusal != comply
        # matched register: neither pole may be wildly longer than the other
        assert 0.5 < len(refusal) / len(comply) < 2.0, (refusal, comply)
    assert len(set(E.REFUSAL_PREFIXES)) == len(E.REFUSAL_PREFIXES)
    assert len(set(E.COMPLIANCE_PREFIXES)) == len(E.COMPLIANCE_PREFIXES)
    # the poles must not overlap — a string appearing on both sides would cancel the contrast
    assert not set(E.REFUSAL_PREFIXES) & set(E.COMPLIANCE_PREFIXES)
