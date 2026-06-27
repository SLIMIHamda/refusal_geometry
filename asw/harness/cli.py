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


def _drefuse_path(cfg):
    from .. import db as dbm

    return (Path(cfg["paths"]["cache_dir"]) / "drefuse"
            / f"{dbm.safe_name(cfg['model']['id'])}.npz")


def _split(bench, cfg, which):
    lo, hi = cfg["splits"]["advbench"][which]
    return bench.prompts()[lo:hi]


def _extract(args) -> int:
    from .. import db as dbm
    from ..data.benchmarks import load_benchmark
    from ..geometry.extract import extract_drefuse, layer_consistency, save_drefuse
    from ..models.loader import load_model, model_commit_hash
    from ..runlog import run_context

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    bench = load_benchmark(args.benchmark, data_dir=cfg["paths"]["data_dir"])
    prompts = _split(bench, cfg, "extract")
    layers = args.layers if args.layers else cfg["model"]["steer_layers"]
    model, tok = load_model(cfg, quant=args.quant)
    mh = model_commit_hash(model)
    with run_context(con, experiment="extract", model_id=cfg["model"]["id"],
                     config=cfg, seed=0, model_hash=mh) as h:
        d = extract_drefuse(model, tok, prompts, layers)
        path = _drefuse_path(cfg)
        save_drefuse(path, d, meta={"model_hash": mh, "layers": layers,
                                    "n_prompts": len(prompts)})
        h["metrics"] = {"layer_consistency": layer_consistency(d), "n_prompts": len(prompts),
                        "layers": layers, "cache": str(path)}
    print(f"[extract] d_refuse for layers {layers} -> {path}")
    return 0


def _geometry_map(args) -> int:
    from .. import db as dbm
    from ..data.benchmarks import load_benchmark
    from ..geometry.extract import load_drefuse
    from ..geometry.projection import anti_alignment_map
    from ..models.loader import load_model, model_commit_hash
    from ..runlog import run_context

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    path = _drefuse_path(cfg)
    if not path.exists():
        print(f"ERROR: no d_refuse cache at {path}; run `asw extract` first", file=sys.stderr)
        return 2
    d = load_drefuse(path)
    bench = load_benchmark(args.benchmark, data_dir=cfg["paths"]["data_dir"])
    prompts = _split(bench, cfg, "projection") if args.benchmark == "advbench" else bench.prompts()[:args.limit]
    model, tok = load_model(cfg, quant=args.quant)
    with run_context(con, experiment="geometry-map", model_id=cfg["model"]["id"],
                     config=cfg, seed=0, model_hash=model_commit_hash(model)) as h:
        amap = anti_alignment_map(model, tok, prompts, d)
        h["metrics"] = {f"layer_{l}": v for l, v in amap.items()}
    for l in sorted(amap):
        v = amap[l]
        print(f"[geometry] layer {l:2d}: <y,d>={v['mean']:+.3f} "
              f"CI=[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]  {v['label']}")
    return 0


def _models_verify(args) -> int:
    from ..models.loader import load_model, verify_model

    cfg = load_config(args.config)
    model, tok = load_model(cfg, quant=args.quant)
    info = verify_model(model, tok, cfg)
    for k, v in info.items():
        print(f"[verify] {k:14s}: {v}")
    if not info["deterministic"]:
        print("[verify] WARNING: non-deterministic generation at T=0", file=sys.stderr)
        return 1
    return 0


def _eval(args) -> int:
    from .. import db as dbm
    from ..data.benchmarks import load_benchmark
    from ..models.loader import load_model, model_commit_hash
    from ..scorers.judge import HFClassifierJudge, RubricJudge
    from .evaluate import evaluate_benchmark
    from .generate import HFGenerator

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    bench = load_benchmark(args.benchmark, data_dir=cfg["paths"]["data_dir"], limit=args.limit)
    model, tok = load_model(cfg, quant=args.quant)
    gen = HFGenerator(model, tok)
    judges = {"rubric": RubricJudge()}
    if args.hf_judge:
        judges["hf_classifier"] = HFClassifierJudge()

    seeds = args.seeds if args.seeds else cfg["seeds"]
    for seed in seeds:
        metrics = evaluate_benchmark(
            con, generator=gen, benchmark=bench, model_id=cfg["model"]["id"],
            config=cfg, seed=seed, judges=judges, decoding=cfg["decoding"],
            results_dir=cfg["paths"]["results_dir"],
            model_revision=cfg["model"].get("revision"),
            model_hash=model_commit_hash(model),
        )
        for key, val in metrics.items():
            if key.startswith("refusal_rate"):
                print(f"[eval] seed={seed} {key} = {val['rate']:.3f} "
                      f"CI=[{val['ci_lo']:.3f},{val['ci_hi']:.3f}] n={val['n']}")
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

    ex = sub.add_parser("extract", help="behavioral-contrast extraction of d_refuse (WS2)")
    ex.add_argument("--config", required=True)
    ex.add_argument("--benchmark", default="advbench")
    ex.add_argument("--layers", type=int, nargs="*", default=None)
    ex.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    ex.set_defaults(func=_extract)

    gm = sub.add_parser("geometry-map", help="anti-alignment projection map per layer (C1)")
    gm.add_argument("--config", required=True)
    gm.add_argument("--benchmark", default="advbench")
    gm.add_argument("--limit", type=int, default=200)
    gm.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    gm.set_defaults(func=_geometry_map)

    mv = sub.add_parser("models-verify", help="load a model and check T=0 determinism (M1)")
    mv.add_argument("--config", required=True)
    mv.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    mv.set_defaults(func=_models_verify)

    ev = sub.add_parser("eval", help="run a benchmark on a model over seeds")
    ev.add_argument("--config", required=True)
    ev.add_argument("--benchmark", required=True)
    ev.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--seeds", type=int, nargs="*", default=None)
    ev.add_argument("--hf-judge", action="store_true")
    ev.set_defaults(func=_eval)

    so = sub.add_parser("score", help="dual-score a responses parquet")
    so.add_argument("--responses", required=True)
    so.add_argument("--hf-judge", action="store_true", help="also run the HF classifier (#2)")
    so.add_argument("--out", default=None, help="write scored parquet here")
    so.set_defaults(func=_score)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
