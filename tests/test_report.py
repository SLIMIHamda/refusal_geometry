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
        {"layer_13": {"mean": -0.15, "label": "anti-aligned", "ci_lo": -0.2, "ci_hi": -0.1, "n": 40},
         "layer_14": {"mean": -0.10, "label": "anti-aligned", "ci_lo": -0.15, "ci_hi": -0.05, "n": 40}})
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


def test_table_ablation_alpha(tmp_path):
    a = tables.table_ablation(load_runs(_seed(tmp_path / "r.sqlite")), "alpha")
    assert set(a["point"]) == {"alpha=2", "alpha=8"}


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
