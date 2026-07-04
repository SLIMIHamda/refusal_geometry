import numpy as np

from asw.geometry.trace import (
    aie_grid, aie_ratio, directional_aie_summary, directional_restore, noise_sigma,
    peak_layer, random_unit_like, refusal_probability,
)


def test_refusal_probability_definition_is_pinned():
    # P = summed softmax mass over the refusal-onset id set at the first generated position.
    logits = np.array([0.0, 10.0, 10.0, 0.0])          # ids 1,2 dominate the softmax
    assert refusal_probability(logits, [1, 2]) > 0.999
    # uniform logits -> P = |set| / vocab
    assert abs(refusal_probability(np.zeros(4), [0, 1]) - 0.5) < 1e-9
    # AIE is the change in exactly this scalar under restoration
    p_corrupt = refusal_probability(np.zeros(4), [0])
    p_restore = refusal_probability(np.array([5.0, 0.0, 0.0, 0.0]), [0])
    assert p_restore - p_corrupt > 0


def test_noise_sigma():
    assert abs(noise_sigma(np.ones((4, 4)) * 2.0) - 0.0) < 1e-9   # zero std
    assert noise_sigma(np.array([[0.0, 2.0], [0.0, -2.0]]), k=3.0) > 0


def test_aie_grid_and_peak():
    grid = [[0.1, 0.2], [0.9, 0.8], [0.3, 0.3]]          # [layers, positions] of P_restore
    aie = aie_grid(grid, p_corrupt=0.1)
    assert np.allclose(aie[1], [0.8, 0.7])
    assert peak_layer(aie) == 1


def test_directional_restore_only_moves_along_d():
    d = np.array([1.0, 0, 0])
    h_corrupt = np.zeros(3)
    h_clean = np.array([3.0, 4.0, 0.0])
    out = directional_restore(h_corrupt, h_clean, d)
    assert np.allclose(out, [3.0, 0.0, 0.0])             # only the d-component restored


def test_directional_restore_equals_full_when_delta_is_along_d():
    d = np.array([0.0, 1.0, 0.0])
    h_corrupt = np.array([2.0, 0.0, 5.0])
    h_clean = np.array([2.0, 7.0, 5.0])                  # delta purely along d
    assert np.allclose(directional_restore(h_corrupt, h_clean, d), h_clean)


def test_directional_restore_batched():
    d = np.array([1.0, 0.0])
    hc = np.array([[0.0, 0.0], [1.0, 1.0]])
    hk = np.array([[2.0, 9.0], [4.0, 9.0]])
    out = directional_restore(hc, hk, d)
    assert np.allclose(out, [[2.0, 0.0], [4.0, 1.0]])


def test_aie_ratio_handles_zero_full():
    r = aie_ratio([0.4, 0.2], [0.8, 0.0])
    assert abs(r[0] - 0.5) < 1e-9 and np.isnan(r[1])


def test_directional_aie_summary():
    s = directional_aie_summary(aie_full=[0.2, 0.8, 0.4],
                                aie_directional=[0.1, 0.6, 0.2],
                                aie_random=[0.0, 0.05, 0.02])
    assert s["peak_layer_full"] == 1 and s["peak_layer_directional"] == 1
    assert 0.0 < s["mean_ratio"] < 1.0
    assert s["mean_random_ratio"] < s["mean_ratio"]      # control carries less than d_refuse


def test_random_unit_like():
    v = random_unit_like(np.zeros(8), seed=1)
    assert v.shape == (8,) and abs(np.linalg.norm(v) - 1) < 1e-9
