import numpy as np

from asw.baselines.defenses import ClassifierFilter, abliteration_reversal, cast_baseline
from asw.harness.generate import ScriptedGenerator


def test_classifier_filter_blocks_flagged_prompts():
    base = ScriptedGenerator(lambda p: f"answer to {p}")
    flt = ClassifierFilter(base, classify_fn=lambda ps: ["bomb" in p for p in ps],
                           refusal_text="NO")
    out = flt.generate(["how to bomb", "hello there"], temperature=0.0, max_new_tokens=4, seed=0)
    assert out == ["NO", "answer to hello there"]


def test_abliteration_reversal_is_unconditional_raw_add():
    d = {13: np.zeros(4), 14: np.zeros(4)}
    w = abliteration_reversal(model=None, tok=None, d_by_layer=d, alpha=2.0)
    assert w.branch_by_layer == {13: "raw_add", 14: "raw_add"}
    assert w.condition is None and w.alpha == 2.0


def test_cast_baseline_is_conditional_raw_add():
    d = {13: np.zeros(4)}
    w = cast_baseline(model=None, tok=None, d_by_layer=d, alpha=1.5,
                      condition="COND", condition_layer=8)
    assert w.branch_by_layer == {13: "raw_add"}
    assert w.condition == "COND" and w.condition_layer == 8
