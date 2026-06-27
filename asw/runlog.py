"""RunRecord context manager: time a run, capture env, write the row on enter+exit.

Usage:
    with run_context(con, experiment="extract", model_id=mid, config=cfg, seed=0) as h:
        ...                     # do work
        h["metrics"] = {"refusal_rate": 0.31}   # stamped into metrics_json
The row is written 'running' on enter and finalised 'completed'/'failed' on exit,
so a crash still leaves a diagnosable record.
"""
from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

from . import db as dbm
from . import repro
from .config import config_hash as _config_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return None


@contextmanager
def run_context(
    con,
    *,
    experiment: str,
    model_id: str,
    config: dict,
    seed: int,
    model_revision: str | None = None,
    model_hash: str | None = None,
    notes: str | None = None,
):
    chash = _config_hash(config)
    run_id = dbm.make_run_id(experiment, model_id, chash, seed)
    record = {
        "run_id": run_id,
        "experiment": experiment,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_hash": model_hash,
        "seed": seed,
        "config_hash": chash,
        "config_json": json.dumps(config, sort_keys=True),
        "status": "running",
        "started_at": _now(),
        "notes": notes,
        **repro.capture_env(),
    }
    dbm.upsert_run(con, record)
    t0 = time.time()
    handle: dict = {"run_id": run_id, "config_hash": chash, "metrics": {}}
    try:
        yield handle
        record["status"] = "completed"
    except BaseException as e:  # noqa: BLE001 - we record then re-raise
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        raise
    finally:
        record["ended_at"] = _now()
        record["wall_clock_s"] = round(time.time() - t0, 3)
        record["metrics_json"] = json.dumps(handle["metrics"], sort_keys=True)
        record["peak_vram_mb"] = _peak_vram_mb()
        dbm.upsert_run(con, record)
