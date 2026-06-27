"""Results spine: a single-file SQLite run index + Parquet per-prompt artifacts.

- `runs` table: one row per (experiment, model, config, seed) unit. This IS the
  reproducibility manifest (Tab A) and the source for every paper table.
- per-prompt rows -> results/<experiment>/<run_id>.parquet (columnar, out of SQLite).

Resume contract: `run_id` is a deterministic hash, so an interrupted Kaggle/RunPod
session re-derives the same id and `run_exists(..., done=True)` skips finished units.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def safe_name(name: str) -> str:
    """Filesystem-safe form of an experiment name (colon etc. are illegal on Windows)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def run_parquet_path(results_dir: str | Path, experiment: str, run_id: str) -> Path:
    return Path(results_dir) / safe_name(experiment) / f"{run_id}.parquet"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id               TEXT PRIMARY KEY,
    experiment           TEXT NOT NULL,
    model_id             TEXT,
    model_revision       TEXT,
    model_hash           TEXT,
    seed                 INTEGER,
    config_hash          TEXT,
    config_json          TEXT,
    git_commit           TEXT,
    git_dirty            INTEGER,
    python_version       TEXT,
    platform             TEXT,
    torch_version        TEXT,
    transformers_version TEXT,
    accelerate_version   TEXT,
    datasets_version     TEXT,
    cuda_version         TEXT,
    gpu_type             TEXT,
    gpu_count            INTEGER,
    host                 TEXT,
    started_at           TEXT,
    ended_at             TEXT,
    wall_clock_s         REAL,
    peak_vram_mb         REAL,
    status               TEXT,
    error                TEXT,
    metrics_json         TEXT,
    notes                TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_lookup
    ON runs(experiment, model_id, config_hash, seed);
"""


def make_run_id(experiment: str, model_id: str, config_hash: str, seed: int) -> str:
    key = f"{experiment}|{model_id}|{config_hash}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _columns(con: sqlite3.Connection) -> list[str]:
    return [r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()]


def run_exists(con: sqlite3.Connection, run_id: str, done: bool = False) -> bool:
    row = con.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return False
    return (not done) or row["status"] == "completed"


def upsert_run(con: sqlite3.Connection, record: dict[str, Any]) -> None:
    """Insert or update a run row by run_id.

    Partial-update semantics: only columns present in `record` are written, so an
    incremental call (e.g. status-only on finalize) never clobbers env/fields captured
    by an earlier call. Unknown keys are ignored; missing columns keep their value.
    """
    if "run_id" not in record:
        raise ValueError("upsert_run requires 'run_id'")
    known = set(_columns(con))
    cols = [c for c in record if c in known]
    vals = [int(v) if isinstance(v, bool) else v for v in (record[c] for c in cols)]
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "run_id")
    sql = f"INSERT INTO runs ({','.join(cols)}) VALUES ({placeholders})"
    if updates:
        sql += f" ON CONFLICT(run_id) DO UPDATE SET {updates}"
    else:
        sql += " ON CONFLICT(run_id) DO NOTHING"
    con.execute(sql, vals)
    con.commit()


def write_prompt_rows(
    results_dir: str | Path, experiment: str, run_id: str, rows: Iterable[dict]
) -> Path:
    """Persist granular per-prompt rows to Parquet (one file per run)."""
    import pandas as pd

    path = run_parquet_path(results_dir, experiment, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(path, index=False)
    return path
