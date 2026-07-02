import math

import numpy as np
import pytest

from asw.eval import metrics as M


def test_clopper_pearson_zero_is_meaningful_upper_bound():
    # "0.0% ASR" must come with a real bound (Briefing 1, 4.3), not read as perfect.
    rate, lo, hi = M.refusal_rate_ci(0, 100)
    assert rate == 0.0 and lo == 0.0
    assert 0.0 < hi < 0.05            # ~0.036 for n=100


def test_clopper_pearson_full():
    rate, lo, hi = M.refusal_rate_ci(100, 100)
    assert rate == 1.0 and hi == 1.0 and lo > 0.95


def test_counts_exclude_unclear():
    assert M.counts_from_labels(["refusal", "comply", "unclear", "refusal"]) == (2, 3)


def test_wilson_sanity():
    lo, hi = M.wilson_ci(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo < 0.41 and 0.59 < hi < 0.61


def test_mean_ci_single_and_multi():
    assert M.mean_ci([0.5]) == (0.5, 0.5, 0.5)
    m, lo, hi = M.mean_ci([0.4, 0.5, 0.6], seed=0)
    assert lo <= m <= hi and abs(m - 0.5) < 1e-9


def test_agreement_raw_and_kappa():
    a = ["refusal", "comply", "refusal", "refusal"]
    b = ["refusal", "comply", "comply", "refusal"]
    r = M.agreement(a, b)
    assert r["n"] == 4 and abs(r["raw_agreement"] - 0.75) < 1e-9


def test_paired_test_detects_difference():
    a = [0.90, 0.80, 0.85, 0.95]
    b = [0.20, 0.30, 0.25, 0.15]
    res = M.paired_test(a, b, kind="t")
    assert res["p_value"] < 0.05 and res["mean_delta"] > 0


# ── Item 4: multiplicity, power, interaction ──────────────────────────────────
def test_benjamini_hochberg_matches_reference():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    q, reject = M.benjamini_hochberg(np.array(p), alpha=0.05)
    # BH q for the smallest = 0.001*5/1 = 0.005; monotone step-up thereafter
    assert abs(q[0] - 0.005) < 1e-9
    assert reject[0] and reject[1] and not reject[4]
    assert (np.diff(q[np.argsort(p)]) >= -1e-12).all()   # non-decreasing in p-order


def test_benjamini_hochberg_all_null():
    q, reject = M.benjamini_hochberg([0.6, 0.7, 0.8])
    assert not reject.any() and (q <= 1.0).all()


def test_power_mde_roundtrip():
    # at n=100, p0=0.5 the minimal detectable difference is ~0.19 (reviewer's ~18-20 pts)
    mde = M.min_detectable_effect(100, p0=0.5)
    assert 0.16 < mde < 0.22
    # required_n at that MDE should be about 100 (inverse consistency)
    assert M.required_n(mde, p0=0.5) <= 100 + 3
    # a smaller 10-point delta needs a substantially larger sample
    assert M.required_n(0.10, p0=0.5) > 300


def test_mean_pvalue():
    assert M.mean_pvalue([-0.15] * 3 + [-0.16, -0.14]) < 0.01          # clearly below zero
    assert M.mean_pvalue(list(np.linspace(-0.2, 0.2, 21))) > 0.5       # symmetric about zero
    assert math.isnan(M.mean_pvalue([0.3]))                            # too few


def test_crossover_interaction_recovers_sign():
    stats = pytest.importorskip("statsmodels")  # runs where statsmodels is installed
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    probs = {("aligned", "none"): 0.5, ("aligned", "raw_add"): 0.6, ("aligned", "project"): 0.85,
             ("anti-aligned", "none"): 0.2, ("anti-aligned", "raw_add"): 0.8,
             ("anti-aligned", "project"): 0.1}
    for geom in ("aligned", "anti-aligned"):
        for pid in range(120):
            for op in ("none", "raw_add", "project"):
                rows.append({"prompt_id": f"{geom}_{pid}", "operator": op, "geometry": geom,
                             "refused": int(rng.random() < probs[(geom, op)])})
    res = M.crossover_interaction(pd.DataFrame(rows))
    # the project x anti-aligned interaction must be strongly negative and significant
    term = next(k for k in res["interactions"] if "project" in k and "anti-aligned" in k)
    assert res["interactions"][term]["coef"] < 0
    assert res["interactions"][term]["p"] < 0.05
