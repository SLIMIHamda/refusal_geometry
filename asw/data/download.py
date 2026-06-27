"""Fetch benchmark datasets from the HF Hub and write them as the harness JSONL schema
(`data/<name>.jsonl`, read by data/benchmarks.py). Host-run: needs `datasets` + network.

The HF ids below are best-effort mirrors (the walledai/* namespace mirrors most safety
benchmarks uniformly). VERIFY each on first download and adjust SPECS if an id/field moved
— the run is recorded with the model+config hash, so a benchmark swap is auditable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# name -> {hf, [config], split, prompt, [category], [answer]}
SPECS: dict[str, dict[str, Any]] = {
    # harmful-refusal axis
    "advbench": {"hf": "walledai/AdvBench", "split": "train", "prompt": "prompt"},
    "harmbench": {"hf": "walledai/HarmBench", "config": "standard", "split": "train",
                  "prompt": "prompt", "category": "category"},
    "strongreject": {"hf": "walledai/StrongREJECT", "split": "train",
                     "prompt": "prompt", "category": "category"},
    # over-refusal axis
    "xstest": {"hf": "walledai/XSTest", "split": "test", "prompt": "prompt",
               "category": "type"},
    "orbench": {"hf": "bench-llm/or-bench", "config": "or-bench-hard-1k", "split": "train",
                "prompt": "prompt", "category": "category"},
    # utility / fluency axes
    "gsm8k": {"hf": "openai/gsm8k", "config": "main", "split": "test",
              "prompt": "question", "answer": "answer"},
    "math500": {"hf": "HuggingFaceH4/MATH-500", "split": "test",
                "prompt": "problem", "answer": "answer", "category": "subject"},
    "wikitext2": {"hf": "wikitext", "config": "wikitext-2-raw-v1", "split": "test",
                  "prompt": "text"},
}


def normalize_row(name: str, raw: dict, spec: dict, idx: int) -> dict:
    """Map a raw dataset record to the harness JSONL row. Pure (no datasets import)."""
    row: dict[str, Any] = {"id": f"{name}_{idx}", "prompt": raw[spec["prompt"]]}
    cat_key = spec.get("category")
    if cat_key and raw.get(cat_key) is not None:
        row["category"] = raw[cat_key]
    ans_key = spec.get("answer")
    if ans_key and ans_key in raw:
        row["answer"] = raw[ans_key]
    return row


def write_jsonl(out_dir: str | Path, name: str, rows: list[dict]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def download_benchmark(name: str, out_dir: str | Path = "data", limit: int | None = None) -> Path:
    if name not in SPECS:
        raise KeyError(f"no download spec for '{name}'; known: {sorted(SPECS)}")
    from datasets import load_dataset

    spec = SPECS[name]
    ds = load_dataset(spec["hf"], spec.get("config"), split=spec["split"])
    rows: list[dict] = []
    for i, raw in enumerate(ds):
        if spec["prompt"] not in raw or not str(raw[spec["prompt"]]).strip():
            continue
        rows.append(normalize_row(name, raw, spec, i))
        if limit is not None and len(rows) >= limit:
            break
    return write_jsonl(out_dir, name, rows)
