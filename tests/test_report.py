import json

import pytest

from asw import db as dbm
from asw.report import tables
from asw.report.build import build_report
from asw.report.load import load_runs, parse_experiment


def _em(rate, k, n):
    return {"refusal_rate.rubric.T0.0": {"rate": rate, "ci_lo": rate - 0.05,
                                         "ci_hi": rate + 0.05, "k": k, "n": n}}


def _seed(db):
    con = dbm.connect(db)

    def add(run_id, experiment, model_id, seed, metrics, config=None):
        dbm.upsert_run(con, {"run_id": run_id, "experiment": experiment, "model_id": model_id,
                             "seed": seed, "status": "completed",
                             "started_at": "2026-01-01T00:00",
                             "config_json": json.dumps(config or {}),
                             "metrics_json": json.dumps(metrics)})

    add("a0", "eval:harmbench", "A", 0, _em(0.3, 6, 20), {"defense": {"kind": "none"}})
    add("a1", "eval:harmbench", "A", 1, _em(0.3, 6, 20), {"defense": {"kind": "none"}})
    add("aw0", "eval:harmbench:wrapper", "A", 0, _em(0.95, 19, 20),
        {"defense": {"kind": "wrapper", "alpha": 8}})
    add("g0", "geometry-map", "A", 0,
        {"layer_13": {"mean": -0.15, "label": "anti-aligned", "ci_lo": -0.2, "ci_hi": -0.1,
                      "p_value": 0.001, "n": 40},
         "layer_14": {"mean": -0.10, "label": "anti-aligned", "ci_lo": -0.15, "ci_hi": -0.05,
                      "p_value": 0.02, "n": 40}})
    add("ab8", "eval:advbench:ablate-alpha:alpha=8", "A", 0, _em(0.9, 18, 20),
        {"ablation": {"axis": "alpha", "point": "alpha=8"}})
    add("ab2", "eval:advbench:ablate-alpha:alpha=2", "A", 0, _em(0.5, 10, 20),
        {"ablation": {"axis": "alpha", "point": "alpha=2"}})
    return db


def test_parse_experiment():
    assert parse_experiment("eval:harmbench") == ("eval", "harmbench", None)
    assert parse_experiment("eval:advbench:ablate-alpha:alpha=8") == \
        ("eval", "advbench", "ablate-alpha:alpha=8")
    assert parse_experiment("geometry-map") == ("geometry-map", None, None)


def test_load_runs_derived_columns(tmp_path):
    runs = load_runs(_seed(tmp_path / "r.sqlite"))
    assert {"kind", "benchmark", "defense", "tag", "is_ablation"}.issubset(runs.columns)
    aw = runs[runs.run_id == "aw0"].iloc[0]
    assert aw["defense"] == "wrapper" and aw["benchmark"] == "harmbench"
    assert bool(runs[runs.run_id == "ab8"].iloc[0]["is_ablation"])


def test_table_refusal_pools_seeds_excludes_ablation(tmp_path):
    runs = load_runs(_seed(tmp_path / "r.sqlite"))
    t = tables.table_refusal(runs)
    none_row = t[(t.model_id == "A") & (t.defense == "none")].iloc[0]
    assert none_row["n"] == 40 and abs(none_row["refusal_rate"] - 0.3) < 1e-9
    assert none_row["seeds"] == 2
    assert "advbench" not in set(t["benchmark"])  # ablation runs excluded from main table


def test_table_refusal_clusters_when_parquet_present(tmp_path):
    """With per-prompt parquet reachable, the CI is prompt-clustered: n = #prompts (not rows) and
    the interval is wider than the naive seed-pooled Clopper-Pearson."""
    con = dbm.connect(tmp_path / "r.sqlite")
    rdir = tmp_path
    for seed in (0, 1):                                  # T=0 -> both seeds identical
        rid = f"none{seed}"
        rows = [{"prompt_id": f"p{i}", "temperature": 0.0,
                 "label_rubric": "refusal" if i < 6 else "comply"} for i in range(20)]
        dbm.write_prompt_rows(rdir, "eval:harmbench", rid, rows)
        dbm.upsert_run(con, {"run_id": rid, "experiment": "eval:harmbench", "model_id": "A",
                             "seed": seed, "status": "completed", "started_at": "2026-01-01",
                             "config_json": json.dumps({"defense": {"kind": "none"}}),
                             "metrics_json": json.dumps(_em(0.3, 6, 20))})
    runs = load_runs(tmp_path / "r.sqlite")
    clustered = tables.table_refusal(runs, results_dir=rdir).iloc[0]
    pooled = tables.table_refusal(runs).iloc[0]          # no results_dir -> CP fallback
    assert clustered["n"] == 20 and pooled["n"] == 40     # clustered n is prompts, not rows
    assert (clustered["ci_hi"] - clustered["ci_lo"]) > (pooled["ci_hi"] - pooled["ci_lo"])


def test_table_geometry(tmp_path):
    g = tables.table_geometry(load_runs(_seed(tmp_path / "r.sqlite")))
    assert list(g["layer"]) == [13, 14] and set(g["label"]) == {"anti-aligned"}
    # BH-FDR q-values computed across the model's layers (Item 4)
    assert {"p_value", "q_value", "significant"}.issubset(g.columns)
    assert g["q_value"].notna().all() and (g["q_value"] >= g["p_value"] - 1e-9).all()


def test_crossover_from_runs_fits_interaction(tmp_path):
    pytest.importorskip("statsmodels")
    import numpy as np
    import pandas as pd

    from asw import db as dbm

    db = tmp_path / "x.sqlite"
    con = dbm.connect(db)
    rdir = tmp_path / "results"
    rng = np.random.default_rng(0)
    probs = {("aligned", "none"): 0.5, ("aligned", "raw_add"): 0.6, ("aligned", "project"): 0.85,
             ("anti-aligned", "none"): 0.2, ("anti-aligned", "raw_add"): 0.8,
             ("anti-aligned", "project"): 0.1}
    geoms = {"AL": "aligned", "AA": "anti-aligned"}

    def add(run_id, experiment, model_id, metrics, config):
        dbm.upsert_run(con, {"run_id": run_id, "experiment": experiment, "model_id": model_id,
                             "seed": 0, "status": "completed", "started_at": "2026-01-01",
                             "config_json": json.dumps(config), "metrics_json": json.dumps(metrics)})

    for mid, geom in geoms.items():
        lbl = "anti-aligned" if geom == "anti-aligned" else "aligned"
        add(f"g_{mid}", "geometry-map", mid,
            {f"layer_{l}": {"mean": -0.1, "label": lbl, "p_value": 0.01, "n": 40} for l in (13, 14)},
            {})
        for op, exp, cfg in [
            ("none", f"eval:harmbench", {"defense": {"kind": "none"}}),
            ("raw_add", f"eval:harmbench:wrapper-raw_add",
             {"defense": {"kind": "wrapper", "force_op": "raw_add"}}),
            ("project", f"eval:harmbench:wrapper-project",
             {"defense": {"kind": "wrapper", "force_op": "project"}}),
        ]:
            rid = f"{mid}_{op}"
            rows = [{"prompt_id": f"p{i}", "temperature": 0.0,
                     "label_rubric": "refusal" if rng.random() < probs[(geom, op)] else "comply"}
                    for i in range(150)]
            dbm.write_prompt_rows(rdir, exp, rid, rows)
            add(rid, exp, mid, _em(0.5, 75, 150), cfg)

    res = tables.crossover_from_runs(load_runs(db), rdir)
    assert res is not None and not res.get("insufficient")
    term = next(k for k in res["interactions"] if "project" in k and "anti-aligned" in k)
    assert res["interactions"][term]["coef"] < 0 and res["interactions"][term]["p"] < 0.05


def _attack_metrics(defense, attack, asr, budgets, mq=1200.0, n=3):
    return {"defense": defense, "attack": attack, "asr": asr, "n_behaviors": n,
            "mean_queries": mq, "asr_at_budgets": {str(b): v for b, v in budgets.items()},
            "per_behavior_success": [{"behavior": "b", "success": True, "queries": 400}]}


def _seed_attacks(db):
    con = dbm.connect(db)

    def add(run_id, experiment, model_id, metrics, started="2026-02-01T00:00"):
        dbm.upsert_run(con, {"run_id": run_id, "experiment": experiment, "model_id": model_id,
                             "seed": 0, "status": "completed", "started_at": started,
                             "config_json": json.dumps({"attack": {"name": experiment}}),
                             "metrics_json": json.dumps(metrics)})

    add("at_none", "attack:gcg:none", "A",
        _attack_metrics("none", "gcg", 0.9, {500: 0.3, 1000: 0.6, 2000: 0.9}))
    add("at_wrap", "attack:gcg-adaptive:wrapper", "A",
        _attack_metrics("wrapper", "gcg-adaptive", 0.3, {500: 0.0, 1000: 0.1, 2000: 0.3}))
    # a later run of the same (defense, attack) pair must win
    add("at_wrap_old", "attack:gcg-adaptive:wrapper", "A",
        _attack_metrics("wrapper", "gcg-adaptive", 0.99, {500: 0.9, 1000: 0.9, 2000: 0.9}),
        started="2026-01-01T00:00")
    return db


def test_table_attacks_latest_per_pair(tmp_path):
    t = tables.table_attacks(load_runs(_seed_attacks(tmp_path / "a.sqlite")))
    assert set(t["attack"]) == {"gcg", "gcg-adaptive"}
    assert {"asr@500", "asr@1000", "asr@2000"}.issubset(t.columns)
    wrap = t[t["defense"] == "wrapper"].iloc[0]
    assert abs(wrap["asr"] - 0.3) < 1e-9                     # latest run, not the 0.99 stale one
    assert abs(wrap["asr@2000"] - 0.3) < 1e-9
    none = t[t["defense"] == "none"].iloc[0]
    assert abs(none["asr@500"] - 0.3) < 1e-9                 # undefended cracks earlier


def test_table_attacks_empty():
    import pandas as pd
    assert tables.table_attacks(pd.DataFrame()).empty


def test_build_report_includes_attacks(tmp_path):
    out = build_report(_seed_attacks(tmp_path / "a.sqlite"), tmp_path / "rep")
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Adversarial robustness" in md
    assert (out / "tables" / "attacks.csv").exists()


def test_table_detector_and_sweep(tmp_path):
    con = dbm.connect(tmp_path / "r.sqlite")
    sweep = [{"tau": 0.4, "tpr": 0.98, "fpr_benign": 0.10, "fpr_over": 0.30},
             {"tau": 0.5, "tpr": 0.95, "fpr_benign": 0.05, "fpr_over": 0.20},
             {"tau": 0.6, "tpr": 0.90, "fpr_benign": 0.02, "fpr_over": 0.12}]
    metrics = {"condition_layer": 15, "tau": 0.5, "train_sep_acc": 0.99, "heldout_auc": 0.97,
               "heldout_tpr": 0.95, "heldout_fpr": 0.05, "xstest_fpr": 0.20,
               "n_heldout_harmful": 100, "n_heldout_benign": 100, "n_xstest": 200,
               "threshold_sweep": sweep}
    dbm.upsert_run(con, {"run_id": "fc0", "experiment": "fit-condition", "model_id": "A",
                         "seed": 0, "status": "completed", "started_at": "2026-01-03",
                         "config_json": "{}", "metrics_json": json.dumps(metrics)})
    runs = load_runs(tmp_path / "r.sqlite")
    t = tables.table_detector(runs)
    row = t.iloc[0]
    assert row["model_id"] == "A" and abs(row["heldout_auc"] - 0.97) < 1e-9
    assert abs(row["xstest_fpr"] - 0.20) < 1e-9        # the over-refusal number
    sw, tau = tables.latest_threshold_sweep(runs)
    assert tau == 0.5 and len(sw) == 3


def test_build_report_includes_detector(tmp_path):
    con = dbm.connect(tmp_path / "r.sqlite")
    metrics = {"condition_layer": 15, "tau": 0.5, "train_sep_acc": 0.99, "heldout_auc": 0.97,
               "heldout_tpr": 0.95, "heldout_fpr": 0.05, "xstest_fpr": 0.20,
               "n_heldout_harmful": 100, "n_xstest": 200,
               "threshold_sweep": [{"tau": 0.5, "tpr": 0.95, "fpr_benign": 0.05, "fpr_over": 0.2}]}
    dbm.upsert_run(con, {"run_id": "fc0", "experiment": "fit-condition", "model_id": "A",
                         "seed": 0, "status": "completed", "started_at": "2026-01-03",
                         "config_json": "{}", "metrics_json": json.dumps(metrics)})
    out = build_report(tmp_path / "r.sqlite", tmp_path / "rep")
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Detector characterization" in md
    assert (out / "tables" / "detector.csv").exists()


def test_disagreements_and_adjudication(tmp_path):
    con = dbm.connect(tmp_path / "r.sqlite")
    rdir = tmp_path
    rows = []
    for i in range(6):
        lr = "refusal" if i < 3 else "comply"
        lh = lr if i not in (2, 4) else ("comply" if lr == "refusal" else "refusal")
        rows.append({"prompt_id": f"p{i}", "prompt": f"q{i}", "response": f"r{i}",
                     "temperature": 0.0, "label_rubric": lr, "label_hf_classifier": lh})
    dbm.write_prompt_rows(rdir, "eval:harmbench", "run0", rows)
    dbm.upsert_run(con, {"run_id": "run0", "experiment": "eval:harmbench", "model_id": "A",
                         "seed": 0, "status": "completed", "started_at": "2026-01-01",
                         "config_json": json.dumps({"defense": {"kind": "none"}}),
                         "metrics_json": json.dumps(_em(0.5, 3, 6))})
    dis = tables.disagreements(load_runs(tmp_path / "r.sqlite"), rdir)
    assert set(dis["prompt_id"]) == {"p2", "p4"}          # only the two disagreements

    import pandas as pd
    labeled = pd.DataFrame({"prompt_id": ["p2", "p4"],
                            "label_rubric": ["refusal", "comply"],
                            "label_hf_classifier": ["comply", "refusal"],
                            "human_label": ["refusal", "refusal"]})   # p2→rubric, p4→hf
    summ = tables.adjudication_summary(labeled)
    assert summ["n"] == 2 and summ["agree_rubric"] == 0.5 and summ["agree_hf_classifier"] == 0.5
    assert summ["winner"] == "tie"


def test_build_report_adjudication_verdict(tmp_path):
    import pandas as pd

    con = dbm.connect(tmp_path / "r.sqlite")
    dbm.write_prompt_rows(tmp_path, "eval:harmbench", "run0",
                          [{"prompt_id": "p0", "prompt": "q", "response": "r", "temperature": 0.0,
                            "label_rubric": "refusal", "label_hf_classifier": "comply"}])
    dbm.upsert_run(con, {"run_id": "run0", "experiment": "eval:harmbench", "model_id": "A",
                         "seed": 0, "status": "completed", "started_at": "2026-01-01",
                         "config_json": json.dumps({"defense": {"kind": "none"}}),
                         "metrics_json": json.dumps(_em(1.0, 1, 1))})
    pd.DataFrame({"prompt_id": ["p0"], "label_rubric": ["refusal"],
                  "label_hf_classifier": ["comply"], "human_label": ["refusal"]}).to_csv(
        tmp_path / "adjudication_labeled.csv", index=False)
    out = build_report(tmp_path / "r.sqlite", tmp_path / "rep")
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Scorer adjudication" in md and "**rubric**" in md   # human sided with rubric
    assert (out / "tables" / "adjudication_disagreements.csv").exists()


def test_table_ablation_alpha(tmp_path):
    a = tables.table_ablation(load_runs(_seed(tmp_path / "r.sqlite")), "alpha")
    assert set(a["point"]) == {"alpha=2", "alpha=8"}


def test_table_validation(tmp_path):
    con = dbm.connect(tmp_path / "v.sqlite")
    metrics = {
        "ablation": {"refusal_base": 0.9, "refusal_ablated": 0.2, "refusal_drop": 0.7,
                     "threshold": 0.4, "passes": True, "n": 100},
        "template_stability_min": 0.93,
        "natural_teacher_forced_cos": {"13": 0.80, "14": 0.82},
        "natural_refusal_cos": {"13": 0.60, "14": 0.62},
        "behavioral_vs_naive_cos": {"13": 0.50, "14": 0.55},
    }
    dbm.upsert_run(con, {"run_id": "v0", "experiment": "validate-drefuse", "model_id": "A",
                         "seed": 0, "status": "completed", "started_at": "2026-01-02",
                         "config_json": "{}", "metrics_json": json.dumps(metrics)})
    v = tables.table_validation(load_runs(tmp_path / "v.sqlite"))
    row = v.iloc[0]
    assert row["model_id"] == "A" and bool(row["ablation_pass"])
    assert abs(row["refusal_drop"] - 0.7) < 1e-9
    assert abs(row["teacher_forced_cos"] - 0.81) < 1e-9
    assert abs(row["template_min_cos"] - 0.93) < 1e-9


def test_build_report_writes_tables_and_md(tmp_path):
    out = build_report(_seed(tmp_path / "r.sqlite"), tmp_path / "report")
    assert (out / "REPORT.md").exists()
    assert (out / "tables" / "geometry.csv").exists()
    assert (out / "tables" / "refusal.csv").exists()
    assert "Anti-alignment map" in (out / "REPORT.md").read_text(encoding="utf-8")


def test_build_report_makes_figures(tmp_path):
    pytest.importorskip("matplotlib")
    out = build_report(_seed(tmp_path / "r.sqlite"), tmp_path / "report")
    assert (out / "figures" / "anti_alignment_map.png").exists()
    assert (out / "figures" / "alpha_tradeoff.png").exists()


def test_build_report_empty_db(tmp_path):
    p = tmp_path / "empty.sqlite"
    dbm.connect(p)  # creates schema, no rows
    out = build_report(p, tmp_path / "rep")
    assert (out / "REPORT.md").exists()
    assert "no runs yet" in (out / "REPORT.md").read_text(encoding="utf-8")
