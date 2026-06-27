"""Config registry: compose YAML configs and assign each a deterministic hash.

A config may declare `_base_` (str or list of paths relative to the configs root);
bases merge left-to-right via deep-merge, then the local body overrides. The resolved
dict is canonicalised (sorted keys, no whitespace) and sha256'd -> `config_hash`.
Every run row stores this hash, so a paper table is exactly `WHERE config_hash = ...`
and no reported number is ever an orphan.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving `_base_` includes relative to `root`."""
    path = Path(path)
    root = Path(root) if root is not None else path.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    resolved: dict[str, Any] = {}
    for b in bases:
        resolved = _deep_merge(resolved, load_config((path.parent / b), root))
    return _deep_merge(resolved, raw)


def canonical(config: dict) -> str:
    """Order-invariant canonical serialization used for hashing."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict, length: int = 12) -> str:
    return hashlib.sha256(canonical(config).encode()).hexdigest()[:length]
