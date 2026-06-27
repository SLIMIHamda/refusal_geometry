"""asw — Activation-Steering Safety Wrapper.

Measurement spine for the refusal-geometry zoo study and adversarial audit.
The Step-1 modules (repro, config, db, runlog, scorers.refusal) are GPU-free so
the whole tracking/scoring spine is testable on a CPU-only authoring machine.
"""
__version__ = "0.1.0"
