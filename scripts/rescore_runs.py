"""Re-apply the refusal rubric to already-stored generations, in place. CPU-only, no GPU.

WHY. `asw score` re-scores a single parquet's `label_rubric`, and `scripts/audit_generations.py`
backfills the *fluency* channel — but nothing re-derives the stored *refusal* numbers a whole run
manifest reports. When the marker list in `scorers/refusal.py` changes (e.g. the 2026-08 advisory
markers that let the rubric see Dolphin's "I strongly advise against ..." declines), two artifacts
go stale together and must move as one:

  * the per-prompt parquet `label_<judge>` column   -> read by the MAIN table (`table_refusal`,
    which is handed `results_dir`) and the fluency channel;
  * the run's `metrics_json` refusal keys in runs.sqlite -> read by the ABLATION / dose-response
    table (`table_ablation`) and by the main table's Clopper-Pearson fallback.

Re-scoring only the parquet leaves a regenerated report internally inconsistent (headline table on
the new scorer, dose-response on the old one). This script updates BOTH from the same re-scored
labels, reusing the eval harness's own `_summarize`, so the result is byte-identical to what a fresh
`asw eval` would have written — no generation needed.

It is a dry run by default: it prints the before/after refusal counts per run and writes nothing
until `--write`. Back up `results/` (or the bundle dir) before a `--write` pass; the parquet's old
labels are regenerable from git history but the sqlite metrics are not.

    python scripts/rescore_runs.py --results-dir results_POC_notebook/results_bundle          # preview
    python scripts/rescore_runs.py --results-dir results_POC_notebook/results_bundle --write   # apply
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # importable from any cwd


def _merge_notes(existing, prov):
    """Fold a provenance dict into any existing notes JSON without clobbering it. Records WHICH
    scorer produced the current numbers — the config hash does not move when only the marker list
    does, so re-scored metrics otherwise carry no scorer identity (Review B provenance gap)."""
    try:
        base = json.loads(existing) if existing else {}
        if not isinstance(base, dict):
            base = {"_prior_note": existing}
    except (ValueError, TypeError):
        base = {"_prior_note": existing}
    base.update(prov)
    return json.dumps(base)


def _judges(names, want_hf):
    """Build the judge objects we can actually re-run. Rubric is always re-scored; the HF
    classifier only when asked (it needs transformers/torch). Any other stored `label_*` column
    is preserved as-is so its metrics still recompute from the stored labels."""
    from asw.scorers.judge import HFClassifierJudge, RubricJudge

    out = {}
    for n in names:
        if n == "rubric":
            out[n] = RubricJudge()
        elif n == "hf_classifier" and want_hf:
            out[n] = HFClassifierJudge()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results",
                    help="dir holding <experiment>/<run_id>.parquet (a bundle dir, or results/)")
    ap.add_argument("--db", default=None, help="runs.sqlite (default: <results-dir>/runs.sqlite)")
    ap.add_argument("--hf-judge", action="store_true",
                    help="also re-run the HF classifier where a label_hf_classifier column exists")
    ap.add_argument("--write", action="store_true",
                    help="apply: rewrite parquet labels + backfill metrics_json (default: dry run)")
    args = ap.parse_args()

    import pandas as pd

    from asw import db as dbm
    from asw.harness.evaluate import _summarize
    from asw.scorers.refusal import REFUSAL_MARKERS, marker_set_version

    results_dir = Path(args.results_dir)
    db_path = Path(args.db) if args.db else results_dir / "runs.sqlite"
    if not db_path.exists():
        print(f"no runs.sqlite at {db_path.resolve()}", file=sys.stderr)
        return 2

    con = dbm.connect(db_path)   # also runs the additive scorer_version migration
    if args.write:
        filled = dbm.backfill_scorer_version(con)   # reconcile runs stamped before the column
        if filled:
            print(f"backfilled scorer_version from notes on {filled} run(s)")
    runs = con.execute(
        "SELECT run_id, experiment, metrics_json, notes FROM runs WHERE status='completed'"
    ).fetchall()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    print(f"{'run':<44} {'judge':<13} {'old_k':>5} {'new_k':>5} {'n':>4} {'delta':>6}")
    print("-" * 84)
    changed_runs = 0
    total_delta = 0
    for r in runs:
        old_metrics = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
        # only eval runs carry refusal metrics; extract/geometry-map/fit-condition do not
        if not any(k.startswith("refusal_rate.") for k in old_metrics):
            continue
        path = dbm.run_parquet_path(results_dir, r["experiment"], r["run_id"])
        if not path.exists():
            print(f"{r['experiment']:<44} {'-':<13} (parquet missing, skipped)")
            continue
        df = pd.read_parquet(path)
        if "response" not in df.columns:
            continue

        judge_names = [c[len("label_"):] for c in df.columns if c.startswith("label_")]
        judges = _judges(judge_names, args.hf_judge)
        prompts = df.get("prompt", pd.Series([""] * len(df))).tolist()
        responses = df["response"].fillna("").tolist()

        run_delta = 0
        for jname, judge in judges.items():
            old_labels = df[f"label_{jname}"].tolist()
            new_labels = judge.label_batch(prompts, responses)
            df[f"label_{jname}"] = new_labels
            old_k = sum(x == "refusal" for x in old_labels)
            new_k = sum(x == "refusal" for x in new_labels)
            delta = new_k - old_k
            run_delta += delta
            name = r["experiment"][:43]
            flag = "  <==" if delta else ""
            print(f"{name:<44} {jname:<13} {old_k:>5} {new_k:>5} {len(df):>4} {delta:>+6}{flag}")

        # recompute the run manifest metrics from the (re-scored) rows, reusing the eval harness
        # so the JSON shape matches a fresh run exactly; preserve any keys _summarize doesn't emit
        temps = sorted(df["temperature"].unique().tolist()) if "temperature" in df.columns else [0.0]
        summary = _summarize(df.to_dict("records"), judge_names, temps, old_metrics.get("task"))
        new_metrics = {**old_metrics, **summary}

        if run_delta:
            changed_runs += 1
            total_delta += run_delta
        if args.write:
            df.to_parquet(path, index=False)
            notes = _merge_notes(r["notes"], {
                "rescored_at": stamp,
                "rescored_by": "scripts/rescore_runs.py",
                "scorer": "+".join(sorted(judges)),
                "marker_set_version": marker_set_version(),
                "n_markers": len(REFUSAL_MARKERS),
            })
            # experiment is NOT NULL: SQLite checks it on the INSERT arm of the upsert before the
            # run_id conflict resolves, so it must be supplied even for a metrics-only update.
            dbm.upsert_run(con, {"run_id": r["run_id"], "experiment": r["experiment"],
                                 "metrics_json": json.dumps(new_metrics), "notes": notes,
                                 "scorer_version": marker_set_version()})

    print("-" * 84)
    verb = "updated" if args.write else "would change"
    print(f"{verb} {changed_runs} run(s); net refusal delta {total_delta:+d}")
    if not args.write:
        print("\ndry run -- nothing written. re-run with --write to apply "
              "(back up the bundle/results dir first).")
    else:
        print("\nparquet labels rewritten + metrics_json backfilled. "
              "regenerate tables with `asw report` to pick up the new numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
