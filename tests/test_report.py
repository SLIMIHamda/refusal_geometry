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
