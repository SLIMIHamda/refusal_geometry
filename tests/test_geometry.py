import numpy as np

from asw.geometry.extract import (
    cosine, layer_consistency, load_drefuse, mean_difference_direction, min_pairwise_cosine,
    naive_dim_direction, save_drefuse,
)
from asw.geometry.projection import (
    anti_alignment_stats, centered_projections, classify_geometry, norm_decomposition,
    projections, random_direction_null,
)


def test_direction_points_native_to_refusal():
    native = np.zeros((5, 4))
    refusal = np.tile([2.0, 0, 0, 0], (5, 1))
    d = mean_difference_direction(native, refusal)
    assert np.allclose(d, [1, 0, 0, 0])
    assert abs(np.linalg.norm(d) - 1) < 1e-9


def test_naive_dim_direction_points_harmless_to_harmful():
    harmless = np.tile([-2.0, 0, 0, 0], (6, 1))
    harmful = np.tile([3.0, 0, 0, 0], (6, 1))
    d = naive_dim_direction(harmful, harmless)
    assert np.allclose(d, [1, 0, 0, 0]) and abs(np.linalg.norm(d) - 1) < 1e-9


def test_min_pairwise_cosine():
    x, y = [1.0, 0], [0.0, 1.0]
    assert abs(min_pairwise_cosine([x, x, x]) - 1.0) < 1e-9   # identical -> 1
    assert abs(min_pairwise_cosine([x, y])) < 1e-9            # orthogonal -> 0
    assert abs(min_pairwise_cosine([x, x, y]) - 0.0) < 1e-9   # min over pairs picks the 0
    assert np.isnan(min_pairwise_cosine([x]))                 # <2 dirs -> nan


def test_cosine():
    assert abs(cosine([1, 0], [1, 0]) - 1) < 1e-9
    assert abs(cosine([1, 0], [-1, 0]) + 1) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9


def test_projection_sign_is_the_headline():
    d = np.array([1.0, 0, 0, 0])
    anti = np.tile([-3.0, 0, 0, 0], (10, 1))      # activations opposed to d_refuse
    aligned = np.tile([5.0, 0, 0, 0], (10, 1))
    assert np.allclose(projections(anti, d), -1)   # <y, d> < 0  == anti-aligned
    assert np.allclose(projections(aligned, d), 1)


def test_classify_anti_aligned():
    vals = -0.15 + np.random.default_rng(0).normal(0, 0.01, 60)
    out = classify_geometry(vals)
    assert out["label"] == "anti-aligned" and out["ci_hi"] < 0
    assert out["p_value"] < 0.001                     # per-layer p-value for BH-FDR (Item 4)


def test_classify_neutral_pvalue_large():
    vals = np.linspace(-0.2, 0.2, 41)                 # symmetric about zero
    out = classify_geometry(vals)
    assert out["label"] == "neutral" and out["p_value"] > 0.5


def test_classify_neutral():
    vals = np.random.default_rng(0).normal(0, 0.1, 120)
    assert classify_geometry(vals)["label"] == "neutral"


def test_classify_aligned():
    vals = 0.2 + np.random.default_rng(1).normal(0, 0.01, 60)
    assert classify_geometry(vals)["label"] == "aligned"


def test_layer_consistency():
    d = {10: np.array([1.0, 0]), 11: np.array([1.0, 0]), 12: np.array([0.0, 1.0])}
    c = layer_consistency(d)
    assert abs(c[11] - 1) < 1e-9 and abs(c[12]) < 1e-9


def test_save_load_drefuse_roundtrip(tmp_path):
    d = {13: np.array([1.0, 0, 0]), 14: np.array([0.0, 1.0, 0])}
    p = tmp_path / "d.npz"
    save_drefuse(p, d, meta={"model": "x"})
    loaded = load_drefuse(p)
    assert set(loaded) == {13, 14} and np.allclose(loaded[13], [1, 0, 0])
    assert (tmp_path / "d.npz.json").exists()


# ── Item 1: confound-controlled anti-alignment measurement ────────────────────
def test_centered_projections():
    acts = np.tile([2.0, 0.0], (5, 1))
    assert np.allclose(centered_projections(acts, [1.0, 0.0], mu_bg=[1.0, 0.0]), 1.0)


def test_norm_decomposition():
    nd = norm_decomposition(np.array([3.0, 0.0, 4.0]), np.array([1.0, 0.0, 0.0]))
    assert abs(nd["along"] - 3) < 1e-9 and abs(nd["residual_norm"] - 4) < 1e-9
    assert abs(nd["norm"] - 5) < 1e-9 and abs(nd["fraction_along"] - 0.6) < 1e-9


def test_random_direction_null_shape():
    acts = np.random.default_rng(0).normal(0, 1, (50, 6))
    null = random_direction_null(acts, shift=np.ones(6), K=200, seed=0)
    assert null.shape == (200,) and np.isfinite(null).all()


def _acts(rng, D, shift_dim, shift_val, n=200, noise=0.1):
    a = rng.normal(0, noise, (n, D))
    a[:, shift_dim] += shift_val
    return a


def test_anti_alignment_stats_flags_negative_shift():
    rng = np.random.default_rng(0)
    d = np.eye(8)[0]
    s = anti_alignment_stats(_acts(rng, 8, 0, -0.4), d, mu_bg=np.zeros(8), K=500, seed=0)
    assert s["label"] == "anti-aligned" and s["mean"] < 0
    assert s["z_score"] < 0 and s["cohens_d"] < 0 and s["norm"]["along"] < 0


def test_anti_alignment_stats_orthogonal_shift_is_neutral():
    # a LARGE shift orthogonal to d must NOT be flagged (the null's whole point)
    rng = np.random.default_rng(1)
    d = np.eye(8)[0]
    s = anti_alignment_stats(_acts(rng, 8, 3, 0.5), d, mu_bg=np.zeros(8), K=500, seed=1)
    assert s["label"] == "neutral"


def test_anti_alignment_stats_positive_shift_is_aligned():
    rng = np.random.default_rng(2)
    d = np.eye(8)[0]
    s = anti_alignment_stats(_acts(rng, 8, 0, 0.4), d, mu_bg=np.zeros(8), K=500, seed=2)
    assert s["label"] == "aligned" and s["mean"] > 0 and s["z_score"] > 0


def test_anti_alignment_stats_cross_model():
    rng = np.random.default_rng(3)
    d = np.eye(8)[0]
    s = anti_alignment_stats(_acts(rng, 8, 0, -0.4), d, mu_bg=np.zeros(8), K=300,
                             d_cross=np.eye(8)[0])
    assert abs(s["cross_model_cos"] - 1.0) < 1e-9 and "cross_model_mean" in s
