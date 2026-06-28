"""Adversarial robustness suite (C5): GCG, PAIR/TAP, multi-turn persona. The crux test —
does causal localization + conditioning degrade gracefully where naive steering collapses?
Each attack runs against any Generator (naive raw-add, system-prompt, the wrapper)."""
from .common import AttackResult, attack_success_rate, judged_comply  # noqa: F401
from .multiturn import PERSONAS, run_multiturn  # noqa: F401
from .pair import PairConfig, run_pair  # noqa: F401
