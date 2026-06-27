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
