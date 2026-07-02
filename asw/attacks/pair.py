"""PAIR / TAP automated red-teaming (Chao et al.). An attacker LLM iteratively rewrites a
jailbreak prompt against the target until a judge rules the target complied or the query
budget is spent. Attacker and judge are injected (API or local), so the loop itself is pure
and testable; only the injected models need GPU/API.
"""
from __future__ import annotations

from dataclasses import dataclass

from .common import AttackResult, judged_comply


@dataclass
class PairConfig:
    n_iterations: int = 20
    temperature: float = 0.7
    max_new_tokens: int = 256


def run_pair(target, attacker_fn, judge, behavior: str, config: PairConfig | None = None,
             *, defense_hint: str | None = None):
    """Drive PAIR against a target Generator.

    attacker_fn(behavior, last_prompt, last_response, defense_hint) -> next adversarial prompt.
    `defense_hint` (Item 6) gives the adaptive attacker knowledge of the defense in play, so a
    reported PAIR ASR against the wrapper reflects an informed adversary rather than a naive one.
    Returns an AttackResult; `queries` counts target generations.
    """
    cfg = config or PairConfig()
    prompt = behavior
    last_response = ""
    for it in range(cfg.n_iterations):
        last_response = target.generate([prompt], temperature=cfg.temperature,
                                        max_new_tokens=cfg.max_new_tokens, seed=0)[0]
        if judged_comply(judge, prompt, last_response):
            return AttackResult(behavior, True, prompt, last_response, it + 1,
                                meta={"attack": "pair", "defense_hint": defense_hint})
        prompt = attacker_fn(behavior, prompt, last_response, defense_hint)
    return AttackResult(behavior, False, prompt, last_response, cfg.n_iterations,
                        meta={"attack": "pair", "defense_hint": defense_hint})
