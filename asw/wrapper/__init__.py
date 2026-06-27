"""Geometry-aware conditional steering wrapper (C4).

condition (harmful-input detector) -> geometry branch (raw-add if anti-aligned, projection
amplification if aligned/neutral) -> steer only flagged rows. The Wrapper implements the
Generator protocol, so it plugs into evaluate_benchmark and the attack harness unchanged.
"""
from .condition import ConditionVector  # noqa: F401
from .steer import branch_for_label, op_project_amplify, op_raw_add  # noqa: F401
from .wrapper import Wrapper  # noqa: F401
