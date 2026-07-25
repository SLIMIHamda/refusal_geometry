"""End-to-end benchmark evaluation: generate -> dual-score -> metrics -> persist.

Wraps everything in `run_context` so the run row (Tab A manifest) and the per-prompt
parquet are written with the config hash, seed, env, and metrics. This is the function
the CLI `eval` subcommand and every later experiment call.

Every response is also passed through the degeneracy scorer (scorers/fluency.py) and the
per-prompt `degenerate` flag is persisted alongside the judge labels. This is not optional
instrumentation: the refusal rubric scores gibberish as `comply`, so without the fluency
channel a broken model and a compliant one produce the same headline number.
"""
from __future__ import annotations

from pathlib import Path

from .. import db as dbm
from ..eval import metrics as M
from ..runlog import run_context
from ..scorers.fluency import fluency_kwargs, is_degenerate
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
    experiment_tag: str | None = None,
) -> dict:
    """Run one (benchmark, model, config, seed) evaluation unit. Returns metrics.

    `experiment_tag` (e.g. a defense name) is appended to the experiment id so runs of the
    same benchmark under different defenses live in distinct manifest rows / parquet dirs.
    """
    experiment = f"eval:{benchmark.name}"
    if experiment_tag:
        experiment = f"{experiment}:{experiment_tag}"
    with run_context(
        con, experiment=experiment, model_id=model_id, config=config, seed=seed,
        model_revision=model_revision, model_hash=model_hash,
    ) as h:
        resume = dbm.run_parquet_path(results_dir, experiment, h["run_id"])
        fkw = fluency_kwargs(config)
        rows: list[dict] = []
        for temp in decoding["temperatures"]:
            gen_rows = run_generation(
                generator, benchmark.examples,
                temperature=temp, max_new_tokens=decoding["max_new_tokens"],
                seed=seed, resume_path=resume,
            )
            for r in gen_rows:
                deg, rule = is_degenerate(r["response"] or "", **fkw)
                r["degenerate"] = int(deg)
                r["fluency_rule"] = rule
                for jname, judge in judges.items():
                    r[f"label_{jname}"] = judge.label(r["prompt"], r["response"])
                rows.append(r)
            dbm.write_prompt_rows(results_dir, experiment, h["run_id"], rows)  # checkpoint

        metrics = _summarize(rows, list(judges), decoding["temperatures"], benchmark.task)
        h["metrics"] = metrics
    return metrics


def _summarize(rows, judge_names, temperatures, task) -> dict:
    """Per-(temperature, judge) refusal rate + CI, degeneracy rate, and dual-scorer agreement.

    Three refusal-related keys per (judge, temperature), and the difference between them is the
    whole point of the fluency channel:

      refusal_rate.<judge>.T<t>         over ALL scored responses — the legacy headline number,
                                        unchanged so existing runs stay comparable. Counts a
                                        destroyed model's gibberish as `comply`.
      refusal_rate_fluent.<judge>.T<t>  over FLUENT responses only — "of the outputs that were
                                        actually language, what fraction refused?". Goes to nan
                                        (n=0) when a run is wholly degenerate, which is the
                                        correct reading: refusal is unmeasurable on a broken
                                        model, not zero.
      fluency.T<t>                      the degeneracy rate itself, with a CI.

    Read the first alone and a 0.00 is ambiguous; read it beside `fluency` and it never is.
    """
    out: dict = {"task": task}
    for temp in temperatures:
        sub = [r for r in rows if r["temperature"] == temp]
        fluent = [r for r in sub if not r.get("degenerate")]
        k_deg = sum(int(bool(r.get("degenerate"))) for r in sub)
        p, lo, hi = M.refusal_rate_ci(k_deg, len(sub))
        out[f"fluency.T{temp}"] = {
            "degenerate_rate": p, "ci_lo": lo, "ci_hi": hi, "k": k_deg, "n": len(sub),
        }
        for jname in judge_names:
            for suffix, src in (("", sub), ("_fluent", fluent)):
                k, n = M.counts_from_labels([r[f"label_{jname}"] for r in src])
                p, lo, hi = M.refusal_rate_ci(k, n)
                out[f"refusal_rate{suffix}.{jname}.T{temp}"] = {
                    "rate": p, "ci_lo": lo, "ci_hi": hi, "k": k, "n": n,
                }
        if len(judge_names) >= 2:
            a = [r[f"label_{judge_names[0]}"] for r in sub]
            b = [r[f"label_{judge_names[1]}"] for r in sub]
            out[f"agreement.T{temp}"] = M.agreement(a, b)
    return out
