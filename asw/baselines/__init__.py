"""Competitive baselines (Briefing 1, Axis D), all Generator-compatible so they share
evaluate_benchmark with the wrapper. Reviewers will demand these; we include them from day one.
"""
from .defenses import (  # noqa: F401
    DEFAULT_SAFETY_PROMPT,
    ClassifierFilter,
    abliteration_reversal,
    cast_baseline,
    system_prompt_defense,
)
