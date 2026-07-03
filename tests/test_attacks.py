import math

from asw.attacks.common import (
    AttackResult, asr_at_budgets, attack_success_rate, judged_comply,
)
from asw.attacks.gcg import AFFIRMATIVE, GCGConfig, condition_penalty, target_string
from asw.attacks.multiturn import PERSONAS, run_multiturn
from asw.attacks.pair import PairConfig, run_pair
from asw.attacks.run import run_attack_suite, suite_metrics
from asw.harness.generate import ScriptedGenerator


class StubJudge:
    def __init__(self, comply_if):
        self.comply_if = comply_if

    def label(self, prompt, response):
        return "comply" if self.comply_if(response) else "refusal"


def test_attack_success_rate():
    rs = [AttackResult("b", True, "p", "r", 1), AttackResult("b", False, "p", "r", 2)]
    assert attack_success_rate(rs) == 0.5
    assert math.isnan(attack_success_rate([]))


def test_judged_comply():
    j = StubJudge(lambda r: "sure" in r)
    assert judged_comply(j, "p", "sure here")
    assert not judged_comply(j, "p", "I cannot")


def test_gcg_helpers():
    assert target_string().startswith(AFFIRMATIVE)
    assert target_string("Step 1:") == "Step 1:"
    assert GCGConfig().suffix_len == 20


def test_condition_penalty_hinge():
    # below threshold -> no penalty; above -> lam*(proj - tau + margin)
    assert condition_penalty(0.1, tau=0.5) == 0.0
    assert abs(condition_penalty(0.8, tau=0.5, lam=2.0) - 0.6) < 1e-9
    assert abs(condition_penalty(0.5, tau=0.5, margin=0.1) - 0.1) < 1e-9   # margin bites early


def test_asr_at_budgets_is_monotone():
    rs = [AttackResult("a", True, "p", "r", 100), AttackResult("b", True, "p", "r", 400),
          AttackResult("c", False, "p", "r", 999)]
    curve = asr_at_budgets(rs, [50, 100, 500, 1000])
    assert curve[50] == 0.0 and abs(curve[100] - 1 / 3) < 1e-9
    assert abs(curve[500] - 2 / 3) < 1e-9 and curve[1000] == curve[500]  # failure never counts


def test_pair_succeeds_after_refinement():
    target = ScriptedGenerator(lambda p: "sure here you go" if "JAILBREAK" in p else "I cannot")
    judge = StubJudge(lambda r: "sure" in r)
    state = {"n": 0}

    def attacker(behavior, last_prompt, last_resp, defense_hint):
        state["n"] += 1
        return last_prompt + (" JAILBREAK" if state["n"] >= 2 else " please")

    res = run_pair(target, attacker, judge, "do bad", PairConfig(n_iterations=5))
    assert res.success and "JAILBREAK" in res.final_prompt and res.queries == 3


def test_pair_passes_defense_hint_to_attacker():
    target = ScriptedGenerator(lambda p: "I cannot")
    judge = StubJudge(lambda r: "sure" in r)
    seen = {}

    def attacker(behavior, last_prompt, last_resp, defense_hint):
        seen["hint"] = defense_hint
        return last_prompt + " x"

    res = run_pair(target, attacker, judge, "do bad", PairConfig(n_iterations=2),
                   defense_hint="wrapper(raw_add) at layers 13-16")
    assert not res.success and seen["hint"] == "wrapper(raw_add) at layers 13-16"
    assert res.meta["defense_hint"] == "wrapper(raw_add) at layers 13-16"


def test_multiturn_runs_all_turns_and_scores():
    target = ScriptedGenerator(lambda p: "sure here")
    judge = StubJudge(lambda r: "sure" in r)
    res = run_multiturn(target, "do bad", judge, persona="dan")
    assert res.success and res.queries == len(PERSONAS["dan"])


# ── attack driver (Step 2): orchestration is pure via an injected gcg_fn ───────
def _stub_gcg(qmap, seen=None):
    def gcg(model, tok, behavior, *, config=None, **kw):
        if seen is not None:
            seen.append((behavior, kw))
        q = qmap[behavior]
        return "!!!SUFFIX", 0.1, q, [(0, 0.5, q // 2), (1, 0.1, q)]   # last history entry = q
    return gcg


def test_run_attack_suite_success_and_budget_curve():
    qmap = {"a": 400, "b": 1500, "c": 3000}
    comply = {"a", "b"}
    target = ScriptedGenerator(lambda p: "sure here" if p.split()[0] in comply else "I cannot")
    judge = StubJudge(lambda r: "sure" in r)
    results, curve = run_attack_suite(None, None, target, judge, ["a", "b", "c"],
                                      budgets=[500, 1000, 2000, 4000],
                                      gcg_fn=_stub_gcg(qmap))
    # queries come from history[-1]; final prompt splices the suffix onto the behaviour
    assert [r.queries for r in results] == [400, 1500, 3000]
    assert results[0].final_prompt == "a !!!SUFFIX"
    assert [r.success for r in results] == [True, True, False]
    # budget curve reads ASR off those query counts (c never succeeds)
    assert abs(curve[500] - 1 / 3) < 1e-9 and abs(curve[1000] - 1 / 3) < 1e-9
    assert abs(curve[2000] - 2 / 3) < 1e-9 and curve[4000] == curve[2000]


def test_run_attack_suite_forwards_adaptive_knobs():
    seen = []
    target = ScriptedGenerator(lambda p: "I cannot")
    judge = StubJudge(lambda r: "sure" in r)
    sentinel_steer = object()
    run_attack_suite(None, None, target, judge, ["x"], budgets=[1000],
                     steer=sentinel_steer, condition="COND", condition_layer=15,
                     tau=0.7, penalty_lambda=0.5, penalty_margin=0.1,
                     gcg_fn=_stub_gcg({"x": 900}, seen))
    _, kw = seen[0]
    assert kw["steer"] is sentinel_steer and kw["condition"] == "COND"
    assert kw["condition_layer"] == 15 and kw["tau"] == 0.7
    assert kw["penalty_lambda"] == 0.5 and kw["penalty_margin"] == 0.1


def test_suite_metrics_is_json_shaped():
    qmap = {"a": 400, "c": 3000}
    comply = {"a"}
    target = ScriptedGenerator(lambda p: "sure here" if p.split()[0] in comply else "no")
    judge = StubJudge(lambda r: "sure" in r)
    results, curve = run_attack_suite(None, None, target, judge, ["a", "c"],
                                      budgets=[500, 4000], gcg_fn=_stub_gcg(qmap))
    m = suite_metrics(results, curve, attack="gcg-adaptive", defense="wrapper")
    assert m["attack"] == "gcg-adaptive" and m["defense"] == "wrapper"
    assert abs(m["asr"] - 0.5) < 1e-9 and m["n_behaviors"] == 2
    assert set(m["asr_at_budgets"]) == {"500", "4000"}        # budget keys are strings (JSON-safe)
    assert abs(m["mean_queries"] - 1700) < 1e-9
    assert m["per_behavior_success"][0] == {"behavior": "a", "success": True, "queries": 400}
