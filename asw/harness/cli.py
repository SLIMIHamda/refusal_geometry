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


def _condition_path(cfg):
    from .. import db as dbm

    return (Path(cfg["paths"]["cache_dir"]) / "condition"
            / f"{dbm.safe_name(cfg['model']['id'])}.npz")


def _geometry_labels_path(cfg):
    from .. import db as dbm

    return (Path(cfg["paths"]["cache_dir"]) / "geometry"
            / f"{dbm.safe_name(cfg['model']['id'])}.json")


def _condition_layer(cfg):
    """Layer the harmful-input detector reads. Configurable; default = mid steering band."""
    sl = cfg["model"]["steer_layers"]
    return cfg["model"].get("condition_layer", sl[len(sl) // 2])


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
        lp = _geometry_labels_path(cfg)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps({str(l): v["label"] for l, v in amap.items()}, indent=2),
                      encoding="utf-8")
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


def _fit_condition(args) -> int:
    from .. import db as dbm
    from ..data.benchmarks import load_benchmark
    from ..geometry.extract import capture_terminal
    from ..models.loader import load_model, model_commit_hash
    from ..runlog import run_context
    from ..wrapper.condition import ConditionVector

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    cl = _condition_layer(cfg)
    harmful = _split(load_benchmark("advbench", data_dir=cfg["paths"]["data_dir"]), cfg, "extract")
    benign = load_benchmark(args.benign, data_dir=cfg["paths"]["data_dir"],
                            limit=len(harmful)).prompts()
    model, tok = load_model(cfg, quant=args.quant)
    with run_context(con, experiment="fit-condition", model_id=cfg["model"]["id"],
                     config=cfg, seed=0, model_hash=model_commit_hash(model)) as h:
        ha = capture_terminal(model, tok, harmful, [cl])[cl]
        ba = capture_terminal(model, tok, benign, [cl])[cl]
        cv = ConditionVector.fit(ha, ba)
        path = _condition_path(cfg)
        cv.save(path)
        acc = 0.5 * float(cv.predict(ha).mean()) + 0.5 * float((~cv.predict(ba)).mean())
        h["metrics"] = {"condition_layer": cl, "train_sep_acc": acc,
                        "n_harmful": len(harmful), "n_benign": len(benign), "cache": str(path)}
    print(f"[fit-condition] layer {cl}: train separation acc={acc:.3f} -> {path}")
    return 0


def _build_generator(cfg, model, tok, defense, alpha):
    """Construct the Generator for a chosen defense, loading cached artifacts as needed."""
    from .generate import HFGenerator

    if defense in (None, "none"):
        return HFGenerator(model, tok)
    if defense == "system_prompt":
        from ..baselines.defenses import system_prompt_defense
        return system_prompt_defense(model, tok)

    from ..geometry.extract import load_drefuse
    dp = _drefuse_path(cfg)
    if not dp.exists():
        raise SystemExit(f"defense '{defense}' needs d_refuse; run `asw extract` first ({dp})")
    d = load_drefuse(dp)

    if defense == "abliteration":
        from ..baselines.defenses import abliteration_reversal
        return abliteration_reversal(model, tok, d, alpha)

    cp = _condition_path(cfg)
    if not cp.exists():
        raise SystemExit(f"defense '{defense}' needs the condition vector; run "
                         f"`asw fit-condition` first ({cp})")
    from ..wrapper.condition import ConditionVector
    cond, cl = ConditionVector.load(cp), _condition_layer(cfg)

    if defense == "cast":
        from ..baselines.defenses import cast_baseline
        return cast_baseline(model, tok, d, alpha, cond, cl)
    if defense == "wrapper":
        return _build_wrapper(cfg, model, tok, d, alpha)
    raise SystemExit(f"unknown defense '{defense}'")


def _load_geometry_amap(cfg):
    lp = _geometry_labels_path(cfg)
    if not lp.exists():
        raise SystemExit(f"need geometry labels; run `asw geometry-map` first ({lp})")
    labels = json.loads(lp.read_text(encoding="utf-8"))
    return {int(k): {"label": v} for k, v in labels.items()}


def _build_wrapper(cfg, model, tok, d, alpha, *, layers=None, use_condition=True):
    """The geometry-aware wrapper, with optional layer-band restriction and condition toggle
    (the two ablation knobs beyond alpha)."""
    from ..wrapper.condition import ConditionVector
    from ..wrapper.wrapper import Wrapper

    amap = _load_geometry_amap(cfg)
    if layers is not None:
        layers = set(layers)
        d = {l: v for l, v in d.items() if l in layers}
        amap = {l: v for l, v in amap.items() if l in layers}
    cond = cl = None
    if use_condition:
        cp = _condition_path(cfg)
        if not cp.exists():
            raise SystemExit(f"wrapper needs the condition vector; run `asw fit-condition` ({cp})")
        cond, cl = ConditionVector.load(cp), _condition_layer(cfg)
    return Wrapper.from_geometry_map(model, tok, d, amap, alpha,
                                     condition=cond, condition_layer=cl)


def _ablation_points(axis, *, alpha, alphas, layer_sets):
    """One ablation axis -> [(label, build_kwargs)] for _build_wrapper. Pure/testable."""
    if axis == "alpha":
        return [(f"alpha={a:g}", {"alpha": a}) for a in alphas]
    if axis == "condition":
        return [("cond=on", {"alpha": alpha, "use_condition": True}),
                ("cond=off", {"alpha": alpha, "use_condition": False})]
    if axis == "layers":
        return [("layers=" + ",".join(map(str, ls)), {"alpha": alpha, "layers": list(ls)})
                for ls in layer_sets]
    raise SystemExit(f"unknown ablation axis '{axis}'")


def _eval(args) -> int:
    from .. import db as dbm
    from ..data.benchmarks import load_benchmark
    from ..models.loader import load_model, model_commit_hash
    from ..scorers.judge import HFClassifierJudge, RubricJudge
    from .evaluate import evaluate_benchmark

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    bench = load_benchmark(args.benchmark, data_dir=cfg["paths"]["data_dir"], limit=args.limit)
    model, tok = load_model(cfg, quant=args.quant)
    gen = _build_generator(cfg, model, tok, args.defense, args.alpha)
    # fold the defense into the config so its hash (manifest) distinguishes the run
    if args.defense and args.defense != "none":
        cfg = {**cfg, "defense": {"kind": args.defense, "alpha": args.alpha}}
    judges = {"rubric": RubricJudge()}
    if args.hf_judge:
        judges["hf_classifier"] = HFClassifierJudge()

    tag = args.defense if args.defense and args.defense != "none" else None
    seeds = args.seeds if args.seeds else cfg["seeds"]
    for seed in seeds:
        metrics = evaluate_benchmark(
            con, generator=gen, benchmark=bench, model_id=cfg["model"]["id"],
            config=cfg, seed=seed, judges=judges, decoding=cfg["decoding"],
            results_dir=cfg["paths"]["results_dir"],
            model_revision=cfg["model"].get("revision"),
            model_hash=model_commit_hash(model), experiment_tag=tag,
        )
        for key, val in metrics.items():
            if key.startswith("refusal_rate"):
                print(f"[eval] seed={seed} {key} = {val['rate']:.3f} "
                      f"CI=[{val['ci_lo']:.3f},{val['ci_hi']:.3f}] n={val['n']}")
    return 0


def _ablate(args) -> int:
    from .. import db as dbm
    from ..data.benchmarks import load_benchmark
    from ..geometry.extract import load_drefuse
    from ..models.loader import load_model, model_commit_hash
    from ..scorers.judge import HFClassifierJudge, RubricJudge
    from .evaluate import evaluate_benchmark

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    bench = load_benchmark(args.benchmark, data_dir=cfg["paths"]["data_dir"], limit=args.limit)
    dp = _drefuse_path(cfg)
    if not dp.exists():
        raise SystemExit(f"ablation needs d_refuse; run `asw extract` first ({dp})")
    model, tok = load_model(cfg, quant=args.quant)
    d = load_drefuse(dp)
    mh = model_commit_hash(model)
    layer_sets = [[int(x) for x in s.split(",")] for s in (args.layer_sets or [])]
    points = _ablation_points(args.axis, alpha=args.alpha, alphas=args.alphas,
                              layer_sets=layer_sets)
    judges = {"rubric": RubricJudge()}
    if args.hf_judge:
        judges["hf_classifier"] = HFClassifierJudge()
    seeds = args.seeds if args.seeds else cfg["seeds"]
    print(f"[ablate] axis={args.axis} points={len(points)} seeds={seeds}")
    for label, kw in points:
        gen = _build_wrapper(cfg, model, tok, d, **kw)
        acfg = {**cfg, "ablation": {"axis": args.axis, "point": label, **kw}}
        for seed in seeds:
            metrics = evaluate_benchmark(
                con, generator=gen, benchmark=bench, model_id=cfg["model"]["id"],
                config=acfg, seed=seed, judges=judges, decoding=cfg["decoding"],
                results_dir=cfg["paths"]["results_dir"],
                model_revision=cfg["model"].get("revision"), model_hash=mh,
                experiment_tag=f"ablate-{args.axis}:{label}")
            for key, val in metrics.items():
                if key.startswith("refusal_rate"):
                    print(f"[ablate] {label:18s} seed={seed} {key}={val['rate']:.3f} "
                          f"CI=[{val['ci_lo']:.3f},{val['ci_hi']:.3f}]")
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

    fc = sub.add_parser("fit-condition", help="fit the harmful-input condition vector (C4)")
    fc.add_argument("--config", required=True)
    fc.add_argument("--benign", default="alpacaeval", help="benign benchmark for the negatives")
    fc.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    fc.set_defaults(func=_fit_condition)

    mv = sub.add_parser("models-verify", help="load a model and check T=0 determinism (M1)")
    mv.add_argument("--config", required=True)
    mv.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    mv.set_defaults(func=_models_verify)

    ev = sub.add_parser("eval", help="run a benchmark on a model over seeds")
    ev.add_argument("--config", required=True)
    ev.add_argument("--benchmark", required=True)
    ev.add_argument("--defense", default="none",
                    choices=["none", "system_prompt", "abliteration", "cast", "wrapper"],
                    help="defense/generator to evaluate (default: undefended model)")
    ev.add_argument("--alpha", type=float, default=8.0, help="steering strength (steered defenses)")
    ev.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--seeds", type=int, nargs="*", default=None)
    ev.add_argument("--hf-judge", action="store_true")
    ev.set_defaults(func=_eval)

    ab = sub.add_parser("ablate", help="sweep one wrapper ablation axis (C4 ablations)")
    ab.add_argument("--config", required=True)
    ab.add_argument("--benchmark", required=True)
    ab.add_argument("--axis", required=True, choices=["alpha", "layers", "condition"])
    ab.add_argument("--alpha", type=float, default=8.0, help="fixed alpha for layers/condition axes")
    ab.add_argument("--alphas", type=float, nargs="*", default=[2, 4, 8, 16])
    ab.add_argument("--layer-sets", nargs="*", default=None,
                    help='for --axis layers, e.g. "13,14" "13,14,15,16"')
    ab.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    ab.add_argument("--limit", type=int, default=None)
    ab.add_argument("--seeds", type=int, nargs="*", default=None)
    ab.add_argument("--hf-judge", action="store_true")
    ab.set_defaults(func=_ablate)

    so = sub.add_parser("score", help="dual-score a responses parquet")
    so.add_argument("--responses", required=True)
    so.add_argument("--hf-judge", action="store_true", help="also run the HF classifier (#2)")
    so.add_argument("--out", default=None, help="write scored parquet here")
    so.set_defaults(func=_score)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
