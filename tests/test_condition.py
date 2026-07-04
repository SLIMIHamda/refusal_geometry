import math

import numpy as np

from asw.wrapper.condition import (
    ConditionVector, roc_auc, threshold_sweep, tpr_fpr,
)


def test_roc_auc_separable_and_reversed():
    pos, neg = [3.0, 4.0, 5.0], [0.0, 1.0, 2.0]
    assert roc_auc(pos, neg) == 1.0            # harmful ranks strictly above benign
    assert roc_auc(neg, pos) == 0.0            # reversed
    assert math.isnan(roc_auc([], neg))


def test_roc_auc_ties_are_half():
    # fully overlapping distributions -> AUC 0.5 (tie-aware ranks)
    assert abs(roc_auc([1.0, 2.0], [1.0, 2.0]) - 0.5) < 1e-9


def test_tpr_fpr_at_threshold():
    tpr, fpr = tpr_fpr([2.0, 3.0, 4.0], [-1.0, 0.0, 5.0], threshold=1.0)
    assert tpr == 1.0 and abs(fpr - 1 / 3) < 1e-9   # one benign (5.0) fires -> FPR 1/3


def test_threshold_sweep_monotone_in_tau():
    sh, sb, so = [2.0, 3.0], [0.0, 1.0], [0.5, 4.0]
    sweep = threshold_sweep(sh, sb, so, taus=[0.0, 2.5, 5.0])
    # every firing rate is non-increasing as tau rises
    for key in ("tpr", "fpr_benign", "fpr_over"):
        vals = [p[key] for p in sweep]
        assert vals == sorted(vals, reverse=True)
    assert sweep[0]["tpr"] == 1.0 and sweep[-1]["tpr"] == 0.0


def test_condition_vector_fit_scores_harmful_higher():
    rng = np.random.default_rng(0)
    harmful = rng.normal(1.0, 0.1, size=(20, 4))
    benign = rng.normal(-1.0, 0.1, size=(20, 4))
    cv = ConditionVector.fit(harmful, benign)
    # held-out characterization should be near-perfect on this cleanly separated toy
    assert roc_auc(cv.score(harmful), cv.score(benign)) > 0.95
