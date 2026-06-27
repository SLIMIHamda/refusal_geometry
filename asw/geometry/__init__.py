"""Refusal geometry: behavioral-contrast extraction of d_refuse (C2), the cross-model
anti-alignment projection map (C1), and the causal trace (C3)."""
from .extract import cosine, extract_drefuse, mean_difference_direction  # noqa: F401
from .projection import classify_geometry, layer_sweep, projections  # noqa: F401
