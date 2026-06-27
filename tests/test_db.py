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
