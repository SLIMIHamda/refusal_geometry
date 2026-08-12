from asw import db as dbm


def test_run_id_deterministic():
    a = dbm.make_run_id("extract", "m", "cfg123", 0)
    b = dbm.make_run_id("extract", "m", "cfg123", 0)
    assert a == b and len(a) == 16
    assert a != dbm.make_run_id("extract", "m", "cfg123", 1)


def test_upsert_idempotent_and_resume(tmp_path):
    con = dbm.connect(tmp_path / "runs.sqlite")
    rid = dbm.make_run_id("extract", "m", "cfg123", 0)
    dbm.upsert_run(con, {
        "run_id": rid, "experiment": "extract", "model_id": "m",
        "config_hash": "cfg123", "seed": 0, "status": "running", "git_dirty": True,
    })
    assert dbm.run_exists(con, rid)
    assert not dbm.run_exists(con, rid, done=True)     # not finished yet -> don't skip

    dbm.upsert_run(con, {
        "run_id": rid, "experiment": "extract", "model_id": "m",
        "config_hash": "cfg123", "seed": 0, "status": "completed",
    })
    assert dbm.run_exists(con, rid, done=True)          # finished -> resume skips it

    n = con.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
    assert n == 1                                        # upsert, not duplicate insert
    assert con.execute("SELECT git_dirty FROM runs WHERE run_id=?", (rid,)).fetchone()[0] == 1


def test_write_prompt_rows(tmp_path):
    import pandas as pd

    path = dbm.write_prompt_rows(tmp_path, "extract", "rid0", [
        {"prompt_id": 0, "refusal": "refusal"},
        {"prompt_id": 1, "refusal": "comply"},
    ])
    assert path.exists()
    assert len(pd.read_parquet(path)) == 2


def test_ensure_columns_adds_scorer_version_to_legacy_db(tmp_path):
    import re
    import sqlite3

    # a real pre-migration DB has every column EXCEPT scorer_version (only that one is new)
    legacy_schema = re.sub(r",\s*\n\s*scorer_version\s+TEXT", "", dbm.SCHEMA)
    assert "scorer_version" not in legacy_schema

    p = tmp_path / "legacy.sqlite"
    con0 = sqlite3.connect(p)
    con0.executescript(legacy_schema)
    con0.commit()
    con0.close()

    con = dbm.connect(p)                            # additive migration runs on connect
    assert "scorer_version" in dbm._columns(con)
    con2 = dbm.connect(p)                           # idempotent: no duplicate column, no error
    assert dbm._columns(con2).count("scorer_version") == 1


def test_backfill_scorer_version_from_notes(tmp_path):
    import json

    con = dbm.connect(tmp_path / "r.sqlite")
    dbm.upsert_run(con, {"run_id": "r1", "experiment": "eval:harmbench", "status": "completed",
                         "notes": json.dumps({"marker_set_version": "abc123"})})
    dbm.upsert_run(con, {"run_id": "g1", "experiment": "geometry-map", "status": "completed"})
    assert dbm.backfill_scorer_version(con) == 1    # only the stamped run
    got = dict(con.execute("SELECT run_id, scorer_version FROM runs").fetchall())
    assert got["r1"] == "abc123"                    # backfilled from the notes stamp
    assert got["g1"] is None                        # non-eval run has no stamp -> left NULL
    assert dbm.backfill_scorer_version(con) == 0    # idempotent
