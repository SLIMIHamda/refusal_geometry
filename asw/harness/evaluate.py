"""End-to-end benchmark evaluation: generate -> dual-score -> metrics -> persist.

Wraps everything in `run_context` so the run row (Tab A manifest) and the per-prompt
parquet are written with the config hash, seed, env, and metrics. This is the function
the CLI `eval` subcommand and every later experiment call.
"""
from __future__ import annotations

from pathlib import Path

from .. import db as dbm
from ..eval import metrics as M
from ..runlog import run_context
from .generate import run_generation


def evaluate_benchmark(
    con,
    *,
    generator,
    benchmark,
    model_id: str,
    config: dict,
    seed: int,
    judges: dict,
    decoding: dict,
    results_dir: str | Path = "results",
    model_revision: str | None = None,
    model_hash: str | None = None,
) -> dict:
    """Run one (benchmark, model, config, seed) evaluation unit. Returns metrics."""
    experiment = f"eval:{benchmark.name}"
    with run_context(
        con, experiment=experiment, model_id=model_id, config=config, seed=seed,
        model_revision=model_revision, model_hash=model_hash,
    ) as h:
        resume = dbm.run_parquet_path(results_dir, experiment, h["run_id"])
        rows: list[dict] = []
        for temp in decoding["temperatures"]:
            gen_rows = run_generation(
                generator, benchmark.examples,
                temperature=temp, max_new_tokens=decoding["max_new_tokens"],
                seed=seed, resume_path=resume,
            )
            for r in gen_rows:
                for jname, judge in judges.items():
                    r[f"label_{jname}"] = judge.label(r["prompt"], r["response"])
                rows.append(r)
            dbm.write_prompt_rows(results_dir, experiment, h["run_id"], rows)  # checkpoint

        metrics = _summarize(rows, list(judges), decoding["temperatures"], benchmark.task)
        h["metrics"] = metrics
    return metrics


def _summarize(rows, judge_names, temperatures, task) -> dict:
    """Per-(temperature, judge) refusal rate + CI, and dual-scorer agreement."""
    out: dict = {"task": task}
    for temp in temperatures:
        sub = [r for r in rows if r["temperature"] == temp]
        for jname in judge_names:
            labels = [r[f"label_{jname}"] for r in sub]
            k, n = M.counts_from_labels(labels)
            p, lo, hi = M.refusal_rate_ci(k, n)
            out[f"refusal_rate.{jname}.T{temp}"] = {
                "rate": p, "ci_lo": lo, "ci_hi": hi, "k": k, "n": n,
            }
        if len(judge_names) >= 2:
            a = [r[f"label_{judge_names[0]}"] for r in sub]
            b = [r[f"label_{judge_names[1]}"] for r in sub]
            out[f"agreement.T{temp}"] = M.agreement(a, b)
    return out
