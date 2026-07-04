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


def table_refusal(runs, *, judge: str = "rubric", temperature=0.0, results_dir=None):
    """Main results: refusal rate per (model, benchmark, defense). Excludes ablation runs.

    The CI is a PROMPT-CLUSTERED bootstrap (resample prompts, not rows) when the per-prompt parquet
    is reachable (`results_dir` given) — responses to the same prompt across seeds/temps are
    correlated (identical at T=0), so pooling replicates as independent trials inflates n and
    understates the interval. Falls back to a pooled Clopper-Pearson when no parquet is available
    (single-seed-equivalent). `n` is the number of independent prompts under clustering."""
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
                         "defense": r["defense"], "seed": r["seed"], "k": m["k"], "n": m["n"],
                         "experiment": r["experiment"], "run_id": r["run_id"]})
    if not rows:
        return _empty(cols)
    df = pd.DataFrame(rows)
    out = []
    for (mid, bench, defn), g in df.groupby(["model_id", "benchmark", "defense"]):
        p, lo, hi, n = _group_ci(g, results_dir, judge=judge, temperature=temperature)
        out.append({"model_id": mid, "benchmark": bench, "defense": defn,
                    "refusal_rate": p, "ci_lo": lo, "ci_hi": hi, "n": n,
                    "seeds": g["seed"].nunique()})
    return pd.DataFrame(out).sort_values(["benchmark", "model_id", "defense"]).reset_index(drop=True)


def _group_ci(g, results_dir, *, judge, temperature):
    """Prompt-clustered bootstrap CI from the per-run parquet when available, else pooled CP."""
    from ..eval import metrics as M

    if results_dir is not None:
        frame = _prompt_frame(g, results_dir, judge=judge, temperature=temperature)
        if frame is not None and not frame.empty:
            return M.cluster_bootstrap_rate_ci(frame, cluster="prompt_id", value="refused")
    return _pool(g)


def _prompt_frame(g, results_dir, *, judge, temperature):
    """One 0/1 `refused` row per (prompt, seed) at `temperature`, concatenated across the group's
    seed-runs. None if no parquet is reachable (triggers the CP fallback)."""
    import pandas as pd

    from .. import db as dbm

    col = f"label_{judge}"
    frames = []
    for _, r in g.iterrows():
        path = dbm.run_parquet_path(results_dir, r["experiment"], r["run_id"])
        if not path.exists():
            continue
        pq = pd.read_parquet(path)
        if col not in pq.columns or "prompt_id" not in pq.columns:
            continue
        pq = pq[pq["temperature"] == temperature]
        frames.append(pd.DataFrame({"prompt_id": pq["prompt_id"].astype(str),
                                    "refused": (pq[col] == "refusal").astype(int)}))
    return pd.concat(frames, ignore_index=True) if frames else None


def table_geometry(runs):
    """Anti-alignment map: projection mean, per-layer p-value, and a BH-FDR q-value computed
    ACROSS layers within each model (Item 4) so labels are read on corrected significance."""
    import numpy as np
    import pandas as pd

    from ..eval.metrics import benjamini_hochberg

    cols = ["model_id", "layer", "projection", "z_score", "cohens_d", "cross_model_cos",
            "p_value", "q_value", "significant", "label"]
    if runs.empty:
        return _empty(cols)
    sub = runs[(runs["kind"] == "geometry-map") & (runs["status"] == "completed")]
    rows = []
    for mid, g in sub.groupby("model_id"):
        r = g.sort_values("started_at").iloc[-1]
        for k, v in r["metrics"].items():
            if k.startswith("layer_") and isinstance(v, dict):
                rows.append({"model_id": mid, "layer": int(k.split("_")[1]),
                             "projection": v.get("mean"), "z_score": v.get("z_score"),
                             "cohens_d": v.get("cohens_d"), "cross_model_cos": v.get("cross_model_cos"),
                             "p_value": v.get("p_value"), "label": v.get("label")})
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


# ── adversarial robustness — ASR vs query budget (C5, Item 6) ─────────────────
def table_attacks(runs):
    """Per (defense, attack): attack success rate + the ASR-at-budget points, from the latest
    `attack:*` run of each pair. Budget columns are named `asr@{budget}`. The honest headline is
    the through-defense / detector-aware ASR, not a static-suffix replay — the `attack` column
    says which was run."""
    import pandas as pd

    base = ["defense", "attack", "asr", "n_behaviors", "mean_queries"]
    if runs.empty:
        return _empty(base)
    sub = runs[(runs["kind"] == "attack") & (runs["status"] == "completed")]
    if sub.empty:
        return _empty(base)
    sub = sub.sort_values("started_at")
    rows, budget_cols = {}, set()
    for _, r in sub.iterrows():
        m = r["metrics"]
        if "asr" not in m:
            continue
        row = {"defense": m.get("defense", r["defense"]), "attack": m.get("attack"),
               "asr": m.get("asr"), "n_behaviors": m.get("n_behaviors"),
               "mean_queries": m.get("mean_queries")}
        for b, v in (m.get("asr_at_budgets") or {}).items():
            col = f"asr@{int(b)}"
            row[col] = v
            budget_cols.add((int(b), col))
        rows[(row["defense"], row["attack"])] = row      # latest wins (sorted by started_at)
    if not rows:
        return _empty(base)
    cols = base + [c for _, c in sorted(budget_cols)]
    df = pd.DataFrame(list(rows.values()))
    for _, c in budget_cols:
        if c not in df:
            df[c] = float("nan")
    return df[cols].sort_values(["attack", "defense"]).reset_index(drop=True)


# ── d_refuse construct validity (Item 2) ──────────────────────────────────────
def table_validation(runs):
    """One row per model from the latest validate-drefuse run: the ablation necessary-condition
    test plus the template / teacher-forced / naive cosine summaries (means over band layers)."""
    import numpy as np
    import pandas as pd

    cols = ["model_id", "refusal_base", "refusal_ablated", "refusal_drop", "ablation_pass",
            "template_min_cos", "teacher_forced_cos", "natural_dim_cos", "vs_naive_cos"]
    if runs.empty:
        return _empty(cols)
    sub = runs[(runs["kind"] == "validate-drefuse") & (runs["status"] == "completed")]
    rows = []
    for mid, g in sub.groupby("model_id"):
        m = g.sort_values("started_at").iloc[-1]["metrics"]
        ab = m.get("ablation", {})

        def _mean(key):
            d = m.get(key)
            return float(np.mean(list(d.values()))) if d else float("nan")

        rows.append({"model_id": mid, "refusal_base": ab.get("refusal_base"),
                     "refusal_ablated": ab.get("refusal_ablated"),
                     "refusal_drop": ab.get("refusal_drop"), "ablation_pass": ab.get("passes"),
                     "template_min_cos": m.get("template_stability_min"),
                     "teacher_forced_cos": _mean("natural_teacher_forced_cos"),
                     "natural_dim_cos": _mean("natural_refusal_cos"),
                     "vs_naive_cos": _mean("behavioral_vs_naive_cos")})
    return pd.DataFrame(rows)[cols] if rows else _empty(cols)


# ── detector characterization (Review B, item 4) ──────────────────────────────
def table_detector(runs):
    """Held-out detector quality per model from the latest fit-condition run: ROC-AUC, TPR/FPR at
    the chosen tau, and — the number reviewers ask for — the FPR on XSTest (over-refusal firing).
    Train separation accuracy is kept only as a reference column."""
    import pandas as pd

    cols = ["model_id", "condition_layer", "tau", "train_sep_acc", "heldout_auc", "heldout_tpr",
            "heldout_fpr", "xstest_fpr", "n_heldout_harmful", "n_xstest"]
    if runs.empty:
        return _empty(cols)
    sub = runs[(runs["kind"] == "fit-condition") & (runs["status"] == "completed")]
    rows = []
    for mid, g in sub.groupby("model_id"):
        m = g.sort_values("started_at").iloc[-1]["metrics"]
        if "heldout_auc" not in m:                 # pre-Item-4 run without characterization
            continue
        rows.append({c: m.get(c) for c in cols if c != "model_id"} | {"model_id": mid})
    return pd.DataFrame(rows)[cols] if rows else _empty(cols)


def latest_threshold_sweep(runs):
    """(sweep_list, tau) from the latest fit-condition run, or (None, None)."""
    if runs.empty:
        return None, None
    sub = runs[(runs["kind"] == "fit-condition") & (runs["status"] == "completed")]
    if sub.empty:
        return None, None
    m = sub.sort_values("started_at").iloc[-1]["metrics"]
    return m.get("threshold_sweep"), m.get("tau")


# ── scorer adjudication (Review B, item 7) ────────────────────────────────────
def disagreements(runs, results_dir, *, a: str = "rubric", b: str = "hf_classifier",
                  temperature=0.0):
    """Every response where scorers `a` and `b` disagree, across eval runs (needs both label
    columns — i.e. eval ran with --hf-judge). This is the set a human adjudicates: kappa says the
    scorers agree, not which is right. Columns: model_id, defense, prompt_id, prompt, response,
    label_<a>, label_<b>."""
    import pandas as pd

    from .. import db as dbm

    cols = ["model_id", "defense", "prompt_id", "prompt", "response", f"label_{a}", f"label_{b}"]
    if runs.empty:
        return _empty(cols)
    sub = runs[(runs["kind"] == "eval") & (runs["status"] == "completed") & (~runs["is_ablation"])]
    ca, cb = f"label_{a}", f"label_{b}"
    frames = []
    for _, r in sub.iterrows():
        path = dbm.run_parquet_path(results_dir, r["experiment"], r["run_id"])
        if not path.exists():
            continue
        pq = pd.read_parquet(path)
        if ca not in pq.columns or cb not in pq.columns:
            continue
        pq = pq[(pq["temperature"] == temperature) & (pq[ca] != pq[cb])]
        if pq.empty:
            continue
        out = pd.DataFrame({"model_id": r["model_id"], "defense": r["defense"],
                            "prompt_id": pq["prompt_id"].astype(str),
                            "prompt": pq.get("prompt", ""), "response": pq.get("response", ""),
                            ca: pq[ca], cb: pq[cb]})
        frames.append(out)
    return (pd.concat(frames, ignore_index=True).drop_duplicates("prompt_id")
            if frames else _empty(cols))


def adjudication_summary(labeled, *, a: str = "rubric", b: str = "hf_classifier"):
    """From a hand-labeled disagreement CSV (a `human_label` column of refusal|comply), report how
    often each scorer sides with the human. Returns None if nothing is usable, else
    {n, agree_<a>, agree_<b>, winner}."""
    import pandas as pd

    ca, cb = f"label_{a}", f"label_{b}"
    if labeled is None or "human_label" not in labeled.columns:
        return None
    df = labeled[labeled["human_label"].isin(["refusal", "comply"])]
    if df.empty or ca not in df.columns or cb not in df.columns:
        return None
    agree_a = float((df[ca] == df["human_label"]).mean())
    agree_b = float((df[cb] == df["human_label"]).mean())
    winner = a if agree_a > agree_b else (b if agree_b > agree_a else "tie")
    return {"n": int(len(df)), f"agree_{a}": agree_a, f"agree_{b}": agree_b, "winner": winner}
