"""CLI entrypoint.

    python -m asw.harness.cli selfcheck --config configs/models/dolphin-llama3-8b.yaml
    python -m asw.harness.cli score --responses results/eval__harmbench/<run>.parquet

`score` runs the dual-scorer protocol over an existing responses parquet (offline:
rubric always; HF classifier with --hf-judge) and reports refusal rate + CI + agreement.
Generation-based `eval` lands with the model loader in Step 3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import repro
from ..config import config_hash, load_config


def _selfcheck(args) -> int:
    cfg = load_config(args.config)
    env = repro.capture_env()
    repro.set_seed(args.seed)
    print(f"[selfcheck] config       : {args.config}")
    print(f"[selfcheck] config_hash  : {config_hash(cfg)}")
    print(f"[selfcheck] model.id     : {cfg.get('model', {}).get('id', '<none>')}")
    print(f"[selfcheck] seeds        : {cfg.get('seeds')}")
    print(f"[selfcheck] git_commit   : {env['git_commit']}  dirty={env['git_dirty']}")
    print(f"[selfcheck] python       : {env['python_version']}  torch={env['torch_version']}")
    print(f"[selfcheck] gpu          : {env['gpu_type']} x{env['gpu_count']}")
    if args.dump:
        Path(args.dump).write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print("[selfcheck] OK")
    return 0


def _score(args) -> int:
    import pandas as pd

    from ..eval import metrics as M
    from ..scorers.judge import HFClassifierJudge, RubricJudge

    df = pd.read_parquet(args.responses)
    if "response" not in df.columns:
        print("ERROR: parquet needs a 'response' column", file=sys.stderr)
        return 2
    prompts = df.get("prompt", pd.Series([""] * len(df))).tolist()
    responses = df["response"].fillna("").tolist()

    judges = {"rubric": RubricJudge()}
    if args.hf_judge:
        judges["hf_classifier"] = HFClassifierJudge()
    labels = {n: j.label_batch(prompts, responses) for n, j in judges.items()}
    for n, ls in labels.items():
        k, total = M.counts_from_labels(ls)
        p, lo, hi = M.refusal_rate_ci(k, total)
        print(f"[score] {n:14s} refusal_rate={p:.3f}  95%CI=[{lo:.3f},{hi:.3f}]  n={total}")
    if len(judges) >= 2:
        names = list(judges)
        ag = M.agreement(labels[names[0]], labels[names[1]])
        print(f"[score] agreement raw={ag['raw_agreement']:.3f} kappa={ag['cohen_kappa']:.3f}")
    if args.out:
        for n, ls in labels.items():
            df[f"label_{n}"] = ls
        df.to_parquet(args.out, index=False)
        print(f"[score] wrote {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="asw")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("selfcheck", help="resolve a config, hash it, capture env")
    sc.add_argument("--config", required=True)
    sc.add_argument("--seed", type=int, default=0)
    sc.add_argument("--dump", default=None)
    sc.set_defaults(func=_selfcheck)

    so = sub.add_parser("score", help="dual-score a responses parquet")
    so.add_argument("--responses", required=True)
    so.add_argument("--hf-judge", action="store_true", help="also run the HF classifier (#2)")
    so.add_argument("--out", default=None, help="write scored parquet here")
    so.set_defaults(func=_score)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
