"""Load the run manifest into a tidy DataFrame with config/metrics parsed and the
experiment id decomposed into (kind, benchmark, tag, defense)."""
from __future__ import annotations

import json

KNOWN_DEFENSES = {"none", "system_prompt", "abliteration", "cast", "wrapper"}


def parse_experiment(exp: str):
    """'eval:advbench:wrapper' -> ('eval','advbench','wrapper'); 'geometry-map' -> (.,None,None).
    The tag may itself contain colons (e.g. 'ablate-alpha:alpha=8')."""
    parts = (exp or "").split(":")
    if parts and parts[0] == "eval":
        bench = parts[1] if len(parts) > 1 else None
        tag = ":".join(parts[2:]) if len(parts) > 2 else None
        return "eval", bench, tag
    return (parts[0] if parts else ""), None, None


def _defense_of(config: dict, tag) -> str:
    config = config or {}
    if config.get("defense"):
        return config["defense"].get("kind", "none")
    if config.get("ablation"):
        return "wrapper"
    if isinstance(tag, str):
        if tag in KNOWN_DEFENSES:
            return tag
        if tag.startswith("ablate-"):
            return "wrapper"
    return "none"


def load_runs(db_path):
    """All runs -> DataFrame with parsed `config`/`metrics` dicts and derived columns
    kind, benchmark, tag, defense, is_ablation."""
    import pandas as pd

    from .. import db as dbm

    con = dbm.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM runs", con)
    if df.empty:
        return df
    df["config"] = df["config_json"].apply(lambda s: json.loads(s) if s else {})
    df["metrics"] = df["metrics_json"].apply(lambda s: json.loads(s) if s else {})
    parsed = [parse_experiment(e) for e in df["experiment"]]
    df["kind"] = [p[0] for p in parsed]
    df["benchmark"] = [p[1] for p in parsed]
    df["tag"] = [p[2] for p in parsed]
    df["defense"] = [_defense_of(c, t) for c, t in zip(df["config"], df["tag"])]
    df["is_ablation"] = [isinstance(t, str) and t.startswith("ablate-") for t in df["tag"]]
    return df
