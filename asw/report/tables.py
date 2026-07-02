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
    """Anti-alignment map: projection mean, per-layer p-value, and a BH-FDR q-value computed
    ACROSS layers within each model (Item 4) so labels are read on corrected significance."""
    import numpy as np
    import pandas as pd

    from ..eval.metrics import benjamini_hochberg

    cols = ["model_id", "layer", "projection", "p_value", "q_value", "significant", "label"]
    if runs.empty:
        return _empty(cols)
    sub = runs[(runs["kind"] == "geometry-map") & (runs["status"] == "completed")]
    rows = []
    for mid, g in sub.groupby("model_id"):
        r = g.sort_values("started_at").iloc[-1]
        for k, v in r["metrics"].items():
            if k.startswith("layer_") and isinstance(v, dict):
                rows.append({"model_id": mid, "layer": int(k.split("_")[1]),
                             "projection": v.get("mean"), "p_value": v.get("p_value"),
                             "label": v.get("label")})
    if not rows:
        return _empty(cols)
    df = pd.DataFrame(rows).sort_values(["model_id", "layer"]).reset_index(drop=True)
    df["q_value"] = np.nan
    for mid, g in df.groupby("model_id"):
        ps = g["p_value"].to_numpy(dtype=float)
        if ps.size and np.isfinite(ps).all():
            q, _ = benjamini_hochberg(ps)
            df.loc[g.index, "q_value"] = q
    df["significant"] = df["q_value"] <= 0.05
    return df[cols]


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


# ── operator x geometry crossover interaction (Item 4 headline) ────────────────
def _operator_of(config, defense):
    """Map a run's defense to an interaction operator label, or None to exclude it. Only the
    clean cells are kept: undefended ('none') and the operator-forced wrapper runs."""
    spec = (config or {}).get("defense") or {}
    kind = spec.get("kind", defense)          # real 'none' runs carry no defense key
    if kind in (None, "none"):
        return "none"
    if kind == "wrapper" and spec.get("force_op"):
        return spec["force_op"]               # raw_add | project
    return None                               # other defenses are not part of the clean 2x3


def _model_geometry(runs):
    """model_id -> 'aligned' | 'anti-aligned' from the latest geometry-map run (modal band label)."""
    sub = runs[(runs["kind"] == "geometry-map") & (runs["status"] == "completed")]
    out = {}
    for mid, g in sub.groupby("model_id"):
        r = g.sort_values("started_at").iloc[-1]
        labels = [v.get("label") for k, v in r["metrics"].items()
                  if k.startswith("layer_") and isinstance(v, dict)]
        out[mid] = "anti-aligned" if labels.count("anti-aligned") > labels.count("aligned") else "aligned"
    return out


def build_interaction_df(runs, results_dir, *, judge="rubric", temperature=0.0):
    """Assemble one row per (prompt, operator, geometry, seed) from the eval-run parquet files."""
    import pandas as pd

    from .. import db as dbm

    geo = _model_geometry(runs)
    sub = runs[(runs["kind"] == "eval") & (runs["status"] == "completed") & (~runs["is_ablation"])]
    col = f"label_{judge}"
    frames = []
    for _, r in sub.iterrows():
        op = _operator_of(r["config"], r["defense"])
        if op is None or r["model_id"] not in geo:
            continue
        path = dbm.run_parquet_path(results_dir, r["experiment"], r["run_id"])
        if not path.exists():
            continue
        pq = pd.read_parquet(path)
        if col not in pq.columns:
            continue
        pq = pq[pq["temperature"] == temperature]
        frames.append(pd.DataFrame({
            "prompt_id": pq["prompt_id"].astype(str),
            "refused": (pq[col] == "refusal").astype(int),
            "operator": op, "geometry": geo[r["model_id"]],
            "seed": r["seed"], "model_id": r["model_id"],
        }))
    cols = ["prompt_id", "refused", "operator", "geometry", "seed", "model_id"]
    return pd.concat(frames, ignore_index=True) if frames else _empty(cols)


def crossover_from_runs(runs, results_dir, *, judge="rubric", temperature=0.0):
    """Fit the operator x geometry logistic interaction from the eval parquet. Returns None with
    no eligible runs, an 'insufficient' dict when the design is not populated (needs 'none' plus a
    forced operator across both geometry classes), else the fitted interaction terms."""
    from ..eval.metrics import crossover_interaction

    df = build_interaction_df(runs, results_dir, judge=judge, temperature=temperature)
    if df.empty:
        return None
    ops, geos = set(df["operator"].unique()), set(df["geometry"].unique())
    if "none" not in ops or len(ops) < 2 or len(geos) < 2:
        return {"insufficient": True, "operators": sorted(ops), "geometry": sorted(geos),
                "n_obs": int(len(df)), "n_models": int(df["model_id"].nunique())}
    res = crossover_interaction(df)
    res["operators"], res["geometry"] = sorted(ops), sorted(geos)
    return res
