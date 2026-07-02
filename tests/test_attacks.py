import math

from asw.attacks.common import (
    AttackResult, asr_at_budgets, attack_success_rate, judged_comply,
)
from asw.attacks.gcg import AFFIRMATIVE, GCGConfig, condition_penalty, target_string
from asw.attacks.multiturn import PERSONAS, run_multiturn
from asw.attacks.pair import PairConfig, run_pair
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
