"""Report layer (M5): regenerate every table and figure from results/runs.sqlite.

Nothing in the paper is hand-transcribed — `asw report` reads the run manifest, pools metrics
across seeds, and writes CSV tables + PNG figures + a REPORT.md index. Pure pandas/matplotlib,
so it runs anywhere (no GPU/torch) and is unit-tested on a synthetic manifest.
"""
from .build import build_report  # noqa: F401
from .load import load_runs  # noqa: F401
