"""Meta-tensor-safe model loading + a reproducibility hash.

Under `device_map="auto"` params are sharded (some pass through the `meta` device during
init), which is exactly the thesis constraint: we never edit weights in place; steering is
applied via forward hooks registered AFTER dispatch (models/hooks.py, Step 8).

`quant_spec` is pure and unit-tested — it encodes the locked precision policy:
geometry/extraction run bf16; int8 is the 70B path (1xA100-80G), nf4 the cheap fallback,
both validated by the 8B calibration gate (see memory: 70B precision decision).
"""
from __future__ import annotations

from typing import Any


def quant_spec(quant: str | None) -> dict[str, Any] | None:
    """Map a quant name to BitsAndBytesConfig kwargs. Pure (no bitsandbytes import)."""
    if quant in (None, "none", "bf16", "fp16"):
        return None
    if quant == "int8":
        return {"load_in_8bit": True}
    if quant == "nf4":
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    raise ValueError(f"unknown quant '{quant}' (use None|int8|nf4)")


def model_commit_hash(model) -> str | None:
    """The HF Hub commit sha that uniquely identifies these weights (for the run row)."""
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def load_model(
    cfg: dict,
    *,
    quant: str | None = None,
    attn_implementation: str = "eager",
    device_map: str = "auto",
):
    """Load (model, tokenizer) for a resolved config. bf16 by default; int8/nf4 via `quant`.

    `attn_implementation="eager"` keeps attention patchable for the causal-trace arm (WS3);
    override to "sdpa" for faster pure generation. Tokenizer is set up for left-padded
    decoder-only batched generation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = cfg["model"]
    dtype = getattr(torch, m.get("dtype", "bfloat16"))
    # CLI --quant wins; otherwise the config may pin its own (e.g. 70B -> int8).
    spec = quant_spec(quant if quant is not None else m.get("quant"))
    qconfig = None
    if spec is not None:
        from transformers import BitsAndBytesConfig

        if spec.get("load_in_4bit"):
            spec = {**spec, "bnb_4bit_compute_dtype": dtype}
        qconfig = BitsAndBytesConfig(**spec)

    tok = AutoTokenizer.from_pretrained(m["id"], revision=m.get("revision"))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        m["id"],
        revision=m.get("revision"),
        torch_dtype=dtype if qconfig is None else None,
        quantization_config=qconfig,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model, tok


def verify_model(model, tok, cfg, prompt: str = "Say hello in exactly one word.") -> dict:
    """M1 check: the model loads and generates deterministically at T=0 (greedy)."""
    from ..harness.generate import HFGenerator

    gen = HFGenerator(model, tok)
    a = gen.generate([prompt], temperature=0.0, max_new_tokens=8, seed=0)[0]
    b = gen.generate([prompt], temperature=0.0, max_new_tokens=8, seed=0)[0]
    return {
        "model_id": cfg["model"]["id"],
        "model_hash": model_commit_hash(model),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "deterministic": a == b,
        "sample": a,
    }
