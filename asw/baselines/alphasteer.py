"""AlphaSteer / calibration-method (Jailbreak Antidote) integration point.

AlphaSteer learns refusal steering under a null-space constraint to preserve utility; it is
the 2026 utility-preserving competitor and the bar for the trade-off curve. Per the risk
register we prefer the authors' released code over a from-scratch reimplementation.

Integrate it as a Generator (a `.generate(prompts, *, temperature, max_new_tokens, seed)`
method) and register it here, so it flows through evaluate_benchmark and the attack runner
exactly like every other defense. If the released code cannot be reproduced on our harness,
fall back to comparing against the authors' reported numbers + our re-implementation, stated
clearly in the paper (risk-register mitigation).
"""
from __future__ import annotations


def alphasteer_baseline(*args, **kwargs):
    raise NotImplementedError(
        "AlphaSteer baseline: wrap the authors' released code as a Generator and register it "
        "here (see module docstring). Tracked as a WS4 task; not auto-reproducible."
    )
