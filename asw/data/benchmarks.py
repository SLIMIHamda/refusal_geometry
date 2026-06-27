"""Uniform benchmark loaders (Briefing 1, Axis B).

Every benchmark resolves to a `Benchmark(name, task, examples)` where each `Example`
has a `prompt` and free-form `meta`. The harness only ever sees this interface, so
HarmBench / StrongREJECT / XSTest / OR-Bench / MMLU / GSM8K / WikiText-2 are
interchangeable downstream.

Loading order for `load_benchmark(name)`:
1. local JSONL at `<data_dir>/<name>.jsonl` (offline-first; the release ships these)
2. a registered HF-`datasets` loader (only if installed + online)

JSONL row schema (only `prompt` is required):
    {"id": <any>, "prompt": <str>, "category": <str|null>, ...rest -> meta}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# canonical benchmark -> task axis (Axis B groups)
TASKS: dict[str, str] = {
    "harmbench": "harmful",
    "strongreject": "harmful",
    "advbench": "harmful",
    "xstest": "over_refusal",
    "orbench": "over_refusal",
    "mmlu": "utility",
    "gsm8k": "utility",
    "math500": "utility",
    "alpacaeval": "utility",
    "wikitext2": "fluency",
}

_HF_LOADERS: dict[str, Callable[..., "Benchmark"]] = {}


@dataclass
class Example:
    id: Any
    prompt: str
    category: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Benchmark:
    name: str
    task: str
    examples: list[Example]

    def __len__(self) -> int:
        return len(self.examples)

    def prompts(self) -> list[str]:
        return [e.prompt for e in self.examples]


def register(name: str, task: str) -> Callable:
    """Register an HF-`datasets`-backed loader for `name` (used when no JSONL exists)."""
    TASKS.setdefault(name, task)

    def deco(fn: Callable[..., Benchmark]) -> Callable[..., Benchmark]:
        _HF_LOADERS[name] = fn
        return fn

    return deco


def from_jsonl(name: str, path: str | Path, limit: int | None = None) -> Benchmark:
    task = TASKS.get(name, "unknown")
    examples: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"{path}:{i} missing required 'prompt' field")
            ex_id = row.pop("id", i)
            prompt = row.pop("prompt")
            category = row.pop("category", None)
            examples.append(Example(id=ex_id, prompt=prompt, category=category, meta=row))
            if limit is not None and len(examples) >= limit:
                break
    return Benchmark(name=name, task=task, examples=examples)


def load_benchmark(
    name: str, data_dir: str | Path = "data", limit: int | None = None
) -> Benchmark:
    """Resolve a benchmark by name: local JSONL first, then a registered HF loader."""
    jsonl = Path(data_dir) / f"{name}.jsonl"
    if jsonl.exists():
        return from_jsonl(name, jsonl, limit=limit)
    if name in _HF_LOADERS:
        return _HF_LOADERS[name](limit=limit)
    raise FileNotFoundError(
        f"No JSONL at {jsonl} and no registered HF loader for '{name}'. "
        f"Provide {name}.jsonl or register a loader."
    )
