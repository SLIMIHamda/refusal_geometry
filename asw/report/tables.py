"""Result tables, built from the run manifest. Refusal rates are POOLED across seeds
(sum k, sum n -> one Wilson CI), which is cleaner than averaging per-seed CIs and matches
the headline numbers the paper reports."""
from __future__ import annotations


def _refusal_key(judge: str, temperature) -> str:
    return f"refusal_rate.{judge}.T{temperature}"


def _empty(cols):
    import pandas as pd

    return pd.DataFrame(columns=cols)


def _pool(group):
    """Pool k/n across a group -> (rate, lo, hi, n)."""
    from ..eval import metrics as M

    k, n = int(group["k"].sum()), int(group["n"].sum())
    p, lo, hi = M.refusal_rate_ci(k, n)
    return p, lo, hi, n


def table_refusal(runs, *, judge: str = "rubric", temperature=0.0):
    """Main results: refusal rate per (model, benchmark, defense), pooled over seeds.
    Excludes ablation runs."""
    import pandas as pd

    cols = ["model_id", "benchmark", "defense", "refusal_rate", "ci_lo", "ci_hi", "n", "seeds"]
    if runs.empty:
        return _empty(cols)
    key = _refusal_key(judge, temperature)
    rows = []
    sub = runs[(runs["kind"] == "eval") & (runs["status"] == "completed") & (~runs["is_ablation"])]
    for _, r in sub.iterrows():
        m = r["metrics"].get(key)
        if m:
            rows.append({"model_id": r["model_id"], "benchmark": r["benchmark"],
                         "defense": r["defense"], "seed": r["seed"], "k": m["k"], "n": m["n"]})
    if not rows:
        return _empty(cols)
    df = pd.DataFrame(rows)
    out = []
    for (mid, bench, defn), g in df.groupby(["model_id", "benchmark", "defense"]):
        p, lo, hi, n = _pool(g)
        out.append({"model_id": mid, "benchmark": bench, "defense": defn,
                    "refusal_rate": p, "ci_lo": lo, "ci_hi": hi, "n": n,
                    "seeds": g["seed"].nunique()})
    return pd.DataFrame(out).sort_values(["benchmark", "model_id", "defense"]).reset_index(drop=True)


def table_geometry(runs):
    """Anti-alignment map: projection mean + label per (model, layer) from the latest
    geometry-map run per model."""
    import pandas as pd

    cols = ["model_id", "layer", "projection", "label"]
    if runs.empty:
        return _empty(cols)
    sub = runs[(runs["kind"] == "geometry-map") & (runs["status"] == "completed")]
    rows = []
    for mid, g in sub.groupby("model_id"):
        r = g.sort_values("started_at").iloc[-1]
        for k, v in r["metrics"].items():
            if k.startswith("layer_") and isinstance(v, dict):
                rows.append({"model_id": mid, "layer": int(k.split("_")[1]),
                             "projection": v.get("mean"), "label": v.get("label")})
    if not rows:
        return _empty(cols)
    return pd.DataFrame(rows).sort_values(["model_id", "layer"]).reset_index(drop=True)


def table_ablation(runs, axis: str, *, judge: str = "rubric", temperature=0.0):
    """Per-point refusal rate for one ablation axis, pooled over seeds."""
    import pandas as pd

    cols = ["benchmark", "point", "refusal_rate", "ci_lo", "ci_hi", "n"]
    if runs.empty:
        return _empty(cols)
    key = _refusal_key(judge, temperature)
    prefix = f"ablate-{axis}:"
    sub = runs[(runs["kind"] == "eval") & (runs["status"] == "completed")
               & runs["tag"].fillna("").str.startswith(prefix)]
    rows = []
    for _, r in sub.iterrows():
        m = r["metrics"].get(key)
        if m:
            rows.append({"benchmark": r["benchmark"], "point": r["tag"][len(prefix):],
                         "seed": r["seed"], "k": m["k"], "n": m["n"]})
    if not rows:
        return _empty(cols)
    df = pd.DataFrame(rows)
    out = []
    for (bench, point), g in df.groupby(["benchmark", "point"]):
        p, lo, hi, n = _pool(g)
        out.append({"benchmark": bench, "point": point, "refusal_rate": p,
                    "ci_lo": lo, "ci_hi": hi, "n": n})
    return pd.DataFrame(out).sort_values(["benchmark", "point"]).reset_index(drop=True)
