"""Model loading + hashing (meta-tensor-safe device_map='auto'). Steering hooks land
in models/hooks.py at the wrapper step (WS5/Step 8)."""
from .loader import load_model, model_commit_hash, quant_spec, verify_model  # noqa: F401
