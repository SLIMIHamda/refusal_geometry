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


def _alpha_cache_path(cfg):
    from .. import db as dbm

    return (Path(cfg["paths"]["cache_dir"]) / "alpha"
            / f"{dbm.safe_name(cfg['model']['id'])}.json")


def _read_selected_alpha(cfg):
    """The pre-registered selected alpha for this model, or None (Item 3)."""
    p = _alpha_cache_path(cfg)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("selected_alpha")


def _write_selected_alpha(cfg, alpha, meta):
    p = _alpha_cache_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"selected_alpha": alpha, **meta}, indent=2), encoding="utf-8")
    return p


def _condition_layer(cfg):
    """Layer the harmful-input detector reads. Configurable; default = mid steering band."""
    sl = cfg["model"]["steer_layers"]
    return cfg["model"].get("condition_layer", sl[len(sl) // 2])


def _split(bench, cfg, which):
    lo, hi = cfg["splits"]["advbench"][which]
    prompts = bench.prompts()
    if len(prompts) < hi:                    # silent truncation would shrink a pre-registered split
        print(f"WARNING: advbench has {len(prompts)} prompts but split '{which}' wants [{lo},{hi}); "
              f"got {len(prompts[lo:hi])}. Download more rows (--limit >= {hi}) to honour the "
              f"pre-registered split.", file=sys.stderr)
    return prompts[lo:hi]


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
    from ..geometry.extract import capture_terminal, load_drefuse
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
    # cross-model direction from an aligned base model's cached d_refuse (Item 1)
    d_cross = None
    if args.base_config:
        bp = _drefuse_path(load_config(args.base_config))
        if not bp.exists():
            raise SystemExit(f"--base-config needs its own d_refuse cache; run extract on it ({bp})")
        d_cross = load_drefuse(bp)
    model, tok = load_model(cfg, quant=args.quant)
    # neutral-corpus mean per layer for centered projections (Item 1); --no-center = legacy cosine
    mu_bg = None
    if not args.no_center:
        neutral = load_benchmark(args.neutral, data_dir=cfg["paths"]["data_dir"],
                                 limit=args.limit).prompts()
        na = capture_terminal(model, tok, neutral, sorted(d))
        mu_bg = {l: na[l].mean(axis=0) for l in d}
    with run_context(con, experiment="geometry-map", model_id=cfg["model"]["id"],
                     config=cfg, seed=0, model_hash=model_commit_hash(model)) as h:
        amap = anti_alignment_map(model, tok, prompts, d, mu_bg=mu_bg, d_cross=d_cross)
        h["metrics"] = {f"layer_{l}": v for l, v in amap.items()}
        lp = _geometry_labels_path(cfg)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps({str(l): v["label"] for l, v in amap.items()}, indent=2),
                      encoding="utf-8")
    for l in sorted(amap):
        v = amap[l]
        extra = (f" z={v['z_score']:+.2f} d={v['cohens_d']:+.2f}" if "z_score" in v else "")
        cross = f" cos_base={v['cross_model_cos']:+.2f}" if "cross_model_cos" in v else ""
        proj_label = "proj_c" if mu_bg is not None else "<y,d>"
        print(f"[geometry] layer {l:2d}: {proj_label}={v['mean']:+.3f} "
              f"CI=[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]  {v['label']}{extra}{cross}")
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
    import numpy as np

    from .. import db as dbm
    from ..data.benchmarks import load_benchmark
    from ..geometry.extract import capture_terminal
    from ..models.loader import load_model, model_commit_hash
    from ..runlog import run_context
    from ..wrapper.condition import ConditionVector, roc_auc, threshold_sweep, tpr_fpr

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    cl = _condition_layer(cfg)
    advbench = load_benchmark("advbench", data_dir=cfg["paths"]["data_dir"])
    harmful = _split(advbench, cfg, "extract")
    # benign: first block trains the detector, a DISJOINT next block is held-out (Item 4)
    benign_all = load_benchmark(args.benign, data_dir=cfg["paths"]["data_dir"],
                                limit=2 * len(harmful)).prompts()
    benign, benign_ho = benign_all[:len(harmful)], benign_all[len(harmful):]
    harmful_ho = _split(advbench, cfg, "eval")                 # disjoint held-out harmful
    try:                                                       # benign-but-scary (over-refusal)
        xstest = load_benchmark("xstest", data_dir=cfg["paths"]["data_dir"],
                                limit=args.xstest_limit).prompts()
    except Exception:                                          # noqa: BLE001 - optional set
        xstest = []
    model, tok = load_model(cfg, quant=args.quant)
    with run_context(con, experiment="fit-condition", model_id=cfg["model"]["id"],
                     config=cfg, seed=0, model_hash=model_commit_hash(model)) as h:
        def score(prompts):
            return (cv.score(capture_terminal(model, tok, prompts, [cl])[cl])
                    if prompts else np.array([]))

        ha = capture_terminal(model, tok, harmful, [cl])[cl]
        ba = capture_terminal(model, tok, benign, [cl])[cl]
        cv = ConditionVector.fit(ha, ba)
        path = _condition_path(cfg)
        cv.save(path)
        acc = 0.5 * float(cv.predict(ha).mean()) + 0.5 * float((~cv.predict(ba)).mean())
        # held-out characterization: AUC + TPR/FPR at tau + XSTest FPR + a threshold sweep (Item 4)
        sh, sb, sx = score(harmful_ho), score(benign_ho), score(xstest)
        auc = roc_auc(sh, sb)
        tpr, fpr = tpr_fpr(sh, sb, cv.threshold)
        xstest_fpr = float((sx > cv.threshold).mean()) if sx.size else float("nan")
        taus = [cv.threshold * s for s in (0.8, 0.9, 1.0, 1.1, 1.2)]
        h["metrics"] = {"condition_layer": cl, "train_sep_acc": acc, "tau": cv.threshold,
                        "n_harmful": len(harmful), "n_benign": len(benign),
                        "n_heldout_harmful": len(harmful_ho), "n_heldout_benign": len(benign_ho),
                        "n_xstest": len(xstest), "heldout_auc": auc, "heldout_tpr": tpr,
                        "heldout_fpr": fpr, "xstest_fpr": xstest_fpr,
                        "threshold_sweep": threshold_sweep(sh, sb, sx, taus), "cache": str(path)}
    print(f"[fit-condition] layer {cl}: train acc={acc:.3f}  held-out AUC={auc:.3f}  "
          f"TPR={tpr:.3f} FPR={fpr:.3f}  XSTest-FPR={xstest_fpr:.3f} -> {path}")
    return 0


def _validate_drefuse(args) -> int:
    """Construct-validity checks for d_refuse (Item 2). Intended on an ALIGNED model, where the
    ablation necessary-condition test (refusal collapses when d_refuse is projected out) is
    meaningful."""
    import numpy as np

    from .. import db as dbm
    from ..data.benchmarks import load_benchmark
    from ..geometry.extract import load_drefuse
    from ..geometry.validate import run_validation
    from ..models.loader import load_model, model_commit_hash
    from ..runlog import run_context
    from ..scorers.judge import RubricJudge

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    dp = _drefuse_path(cfg)
    if not dp.exists():
        raise SystemExit(f"validate-drefuse needs d_refuse; run `asw extract` first ({dp})")
    d = load_drefuse(dp)
    layers = [l for l in (args.layers or cfg["model"]["steer_layers"]) if l in d]
    harmful = _split(load_benchmark("advbench", data_dir=cfg["paths"]["data_dir"]), cfg, "eval")
    harmless = load_benchmark(args.benign, data_dir=cfg["paths"]["data_dir"],
                              limit=len(harmful)).prompts()
    threshold = cfg.get("validation", {}).get("ablation_drop_threshold", 0.40)
    model, tok = load_model(cfg, quant=args.quant)
    with run_context(con, experiment="validate-drefuse", model_id=cfg["model"]["id"],
                     config=cfg, seed=0, model_hash=model_commit_hash(model)) as h:
        m = run_validation(model, tok, d, harmful=harmful, harmless=harmless, layers=layers,
                           judge=RubricJudge(), decoding=cfg["decoding"], threshold=threshold)
        h["metrics"] = m
    ab = m["ablation"]
    print(f"[validate] ablation: refusal {ab['refusal_base']:.3f} -> {ab['refusal_ablated']:.3f} "
          f"(drop {ab['refusal_drop']:+.3f} vs threshold {ab['threshold']:.2f})  "
          f"{'PASS' if ab['passes'] else 'FAIL'}")
    print(f"[validate] template stability : min pairwise cos = {m['template_stability_min']:+.3f}")
    if m.get("natural_teacher_forced_cos"):
        mt = float(np.mean([v for v in m["natural_teacher_forced_cos"].values()]))
        print(f"[validate] teacher-forced     : mean cos(d_forced, d_tf) = {mt:+.3f} "
              f"({m['natural_teacher_forced_counts']})")
    else:
        print(f"[validate] teacher-forced     : too few spontaneous refusals "
              f"({m.get('natural_teacher_forced_counts')})")
    if m["natural_refusal_cos"]:
        mn = float(np.mean([v for v in m["natural_refusal_cos"].values()]))
        print(f"[validate] natural (DIM)      : mean cos(d_forced, d_natural) = {mn:+.3f} "
              f"({m['natural_refusal_counts']})")
    else:
        print(f"[validate] natural (DIM)      : too few spontaneous refusals/complies "
              f"({m['natural_refusal_counts']})")
    vn = float(np.mean([v for v in m["behavioral_vs_naive_cos"].values()]))
    print(f"[validate] behavioral vs naive : mean cos = {vn:+.3f}")
    return 0


def _build_generator(cfg, model, tok, defense, alpha, force_op=None):
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
        return _build_wrapper(cfg, model, tok, d, alpha, force_op=force_op)
    raise SystemExit(f"unknown defense '{defense}'")


def _load_geometry_amap(cfg):
    lp = _geometry_labels_path(cfg)
    if not lp.exists():
        raise SystemExit(f"need geometry labels; run `asw geometry-map` first ({lp})")
    labels = json.loads(lp.read_text(encoding="utf-8"))
    return {int(k): {"label": v} for k, v in labels.items()}


def _build_wrapper(cfg, model, tok, d, alpha, *, layers=None, use_condition=True, force_op=None):
    """The geometry-aware wrapper, with optional layer-band restriction, condition toggle (the
    two ablation knobs beyond alpha), and a forced operator (for the crossover interaction)."""
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
    return Wrapper.from_geometry_map(model, tok, d, amap, alpha, force_op=force_op,
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
    force_op = getattr(args, "force_op", None)
    if force_op and args.defense != "wrapper":
        raise SystemExit("--force-op only applies to --defense wrapper")
    gen = _build_generator(cfg, model, tok, args.defense, args.alpha, force_op=force_op)
    # pre-registration guard (Item 3): a steered headline run must use the frozen selected alpha
    if args.defense in ("wrapper", "cast", "abliteration") and not getattr(args, "force_alpha", False):
        sel = _read_selected_alpha(cfg)
        if sel is not None and abs(float(sel) - args.alpha) > 1e-9:
            print(f"WARNING: alpha={args.alpha} != pre-registered selected alpha {sel} "
                  f"({_alpha_cache_path(cfg)}); alpha must be frozen on the tuning split before "
                  f"headline runs (pass --force-alpha to silence).", file=sys.stderr)
    # fold the defense (and forced operator) into the config so its hash distinguishes the run
    if args.defense and args.defense != "none":
        spec = {"kind": args.defense, "alpha": args.alpha}
        if force_op:
            spec["force_op"] = force_op
        cfg = {**cfg, "defense": spec}
    judges = {"rubric": RubricJudge()}
    if args.hf_judge:
        judges["hf_classifier"] = HFClassifierJudge()

    tag = None
    if args.defense and args.defense != "none":
        tag = f"wrapper-{force_op}" if force_op else args.defense
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
    # alpha pre-registration (Item 3): with --select-over, also score the over-refusal benchmark per
    # alpha and freeze the argmax of (harmful-refusal - over-refusal) to cache/alpha/<model>.json.
    select = args.axis == "alpha" and args.select_over
    over_bench = (load_benchmark(args.select_over, data_dir=cfg["paths"]["data_dir"],
                                 limit=args.limit) if select else None)
    rt_key = f"refusal_rate.rubric.T{cfg['decoding']['temperatures'][0]}"
    sel_points = []
    print(f"[ablate] axis={args.axis} points={len(points)} seeds={seeds}"
          + (f" select-over={args.select_over}" if select else ""))
    for label, kw in points:
        gen = _build_wrapper(cfg, model, tok, d, **kw)
        acfg = {**cfg, "ablation": {"axis": args.axis, "point": label, **kw}}
        harmful_rate = None
        for seed in seeds:
            metrics = evaluate_benchmark(
                con, generator=gen, benchmark=bench, model_id=cfg["model"]["id"],
                config=acfg, seed=seed, judges=judges, decoding=cfg["decoding"],
                results_dir=cfg["paths"]["results_dir"],
                model_revision=cfg["model"].get("revision"), model_hash=mh,
                experiment_tag=f"ablate-{args.axis}:{label}")
            if seed == seeds[0]:
                harmful_rate = metrics.get(rt_key, {}).get("rate")
            for key, val in metrics.items():
                if key.startswith("refusal_rate"):
                    print(f"[ablate] {label:18s} seed={seed} {key}={val['rate']:.3f} "
                          f"CI=[{val['ci_lo']:.3f},{val['ci_hi']:.3f}]")
        if select:
            om = evaluate_benchmark(
                con, generator=gen, benchmark=over_bench, model_id=cfg["model"]["id"],
                config={**acfg, "select_over": args.select_over}, seed=seeds[0], judges=judges,
                decoding=cfg["decoding"], results_dir=cfg["paths"]["results_dir"],
                model_revision=cfg["model"].get("revision"), model_hash=mh,
                experiment_tag=f"ablate-alpha-over:{label}")
            sel_points.append({"alpha": kw["alpha"], "refusal_harmful": harmful_rate,
                               "refusal_over": om.get(rt_key, {}).get("rate")})
    if select and sel_points:
        from ..eval.metrics import select_alpha
        chosen = select_alpha(sel_points)
        path = _write_selected_alpha(cfg, chosen, {"tuning_harmful": args.benchmark,
                                                   "tuning_over": args.select_over,
                                                   "objective": "refusal_harmful - refusal_over",
                                                   "points": sel_points})
        print(f"[ablate] pre-registered selected alpha = {chosen} (max harmful-refusal − "
              f"over-refusal on {args.benchmark}/{args.select_over}) -> {path}")
    return 0


def _attack(args) -> int:
    """Adversarial robustness suite (C5, Item 6). Attacks a defended Generator with GCG
    (static / through-defense / detector-aware) or the cheap multi-turn demo, and records ASR +
    the ASR-vs-budget curve to the manifest under experiment `attack:<attack>:<defense>`."""
    from .. import db as dbm
    from ..attacks.common import asr_at_budgets
    from ..attacks.gcg import GCGConfig
    from ..attacks.run import run_attack_suite, suite_metrics
    from ..data.benchmarks import load_benchmark
    from ..models.loader import load_model, model_commit_hash
    from ..runlog import run_context
    from ..scorers.judge import RubricJudge
    from ..wrapper.condition import ConditionVector
    from ..wrapper.wrapper import Wrapper

    cfg = load_config(args.config)
    con = dbm.connect(cfg["paths"]["results_db"])
    acfg = cfg.get("attack", {})

    # behaviour set: pre-registered AdvBench eval split (disjoint from extract/projection), or
    # the first N of any other benchmark.
    bench = load_benchmark(args.behaviors, data_dir=cfg["paths"]["data_dir"])
    behaviors = _split(bench, cfg, "eval") if args.behaviors == "advbench" else bench.prompts()
    if args.limit:
        behaviors = behaviors[:args.limit]

    force_op = getattr(args, "force_op", None)
    if force_op and args.defense != "wrapper":
        raise SystemExit("--force-op only applies to --defense wrapper")

    model, tok = load_model(cfg, quant=args.quant)
    gen = _build_generator(cfg, model, tok, args.defense, args.alpha, force_op=force_op)
    judge = RubricJudge()                       # pre-registered primary success judge
    budgets = args.budgets or acfg.get("budgets") or None
    gc = None

    if args.attack == "multiturn":
        from ..attacks.multiturn import run_multiturn
        results = [run_multiturn(gen, b, judge) for b in behaviors]
        curve = asr_at_budgets(results, budgets or [1, 2, 3])   # multi-turn 'queries' = turns
    elif args.attack == "pair":
        raise SystemExit("attack 'pair' needs an attacker LLM; wire run_pair with an "
                         "attacker_fn in the notebook/API (not exposed on the CLI).")
    else:                                        # gcg | gcg-adaptive | gcg-detector
        suffix_len = acfg.get("suffix_len", GCGConfig.suffix_len)
        gc = GCGConfig(
            n_steps=args.n_steps or acfg.get("n_steps", GCGConfig.n_steps),
            search_width=args.search_width or acfg.get("search_width", GCGConfig.search_width),
            topk=args.topk or acfg.get("topk", GCGConfig.topk),
            suffix_len=suffix_len, init_suffix=" ".join(["!"] * suffix_len))
        steer = condition = condition_layer = None
        tau = penalty_lambda = penalty_margin = 0.0
        if args.attack == "gcg-adaptive":
            if not isinstance(gen, Wrapper):
                raise SystemExit("gcg-adaptive attacks THROUGH a steered defense; use "
                                 "--defense {wrapper,cast,abliteration}")
            steer = gen.steer_context()         # optimise against the steered model
        elif args.attack == "gcg-detector":
            cp = _condition_path(cfg)
            if not cp.exists():
                raise SystemExit(f"gcg-detector needs the condition vector; run "
                                 f"`asw fit-condition` first ({cp})")
            condition, condition_layer = ConditionVector.load(cp), _condition_layer(cfg)
            tau = condition.threshold           # tau is the detector's own firing threshold
            penalty_lambda = (args.penalty_lambda if args.penalty_lambda is not None
                              else acfg.get("penalty_lambda", 1.0))
            penalty_margin = (args.penalty_margin if args.penalty_margin is not None
                              else acfg.get("penalty_margin", 0.0))
        results, curve = run_attack_suite(
            model, tok, gen, judge, behaviors, config=gc,
            steer=steer, condition=condition, condition_layer=condition_layer,
            tau=tau, penalty_lambda=penalty_lambda, penalty_margin=penalty_margin,
            budgets=budgets, temperature=cfg["decoding"]["temperatures"][0],
            max_new_tokens=cfg["decoding"]["max_new_tokens"], seed=args.seed)

    metrics = suite_metrics(results, curve, attack=args.attack, defense=args.defense)

    # fold attack provenance + defense into the config so the config hash distinguishes the run
    grid = sorted(int(k) for k in metrics["asr_at_budgets"])
    prov = {"name": args.attack, "budgets": grid}
    if gc is not None:
        prov.update(n_steps=gc.n_steps, search_width=gc.search_width, topk=gc.topk,
                    suffix_len=gc.suffix_len)
        if args.attack == "gcg-detector":
            prov.update(penalty_lambda=penalty_lambda, penalty_margin=penalty_margin)
    spec = {"kind": args.defense, "alpha": args.alpha}
    if force_op:
        spec["force_op"] = force_op
    rcfg = {**cfg, "attack": prov, "defense": spec}

    experiment = f"attack:{args.attack}:{args.defense}"
    with run_context(con, experiment=experiment, model_id=cfg["model"]["id"],
                     config=rcfg, seed=args.seed, model_hash=model_commit_hash(model)) as h:
        h["metrics"] = metrics
        dbm.write_prompt_rows(cfg["paths"]["results_dir"], experiment, h["run_id"],
                              [{"behavior": r.behavior, "success": r.success, "queries": r.queries,
                                "final_prompt": r.final_prompt, "response": r.response}
                               for r in results])
    print(f"[attack] {args.attack} vs {args.defense}: ASR={metrics['asr']:.3f} "
          f"n={metrics['n_behaviors']}  ASR@budget={metrics['asr_at_budgets']}")
    return 0


def _report(args) -> int:
    from ..report.build import build_report

    cfg = load_config(args.config)
    out = build_report(cfg["paths"]["results_db"], args.out,
                       judge=args.judge, temperature=args.temperature)
    print(f"[report] wrote {out / 'REPORT.md'} (+ tables/ + figures/)")
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
    gm.add_argument("--neutral", default="orbench",
                    help="neutral corpus for the centered-projection baseline mu_bg (Item 1)")
    gm.add_argument("--no-center", action="store_true",
                    help="legacy uncentered cosine classification (skip the mu_bg baseline)")
    gm.add_argument("--base-config", default=None,
                    help="aligned base model config for the cross-model projection (Item 1)")
    gm.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    gm.set_defaults(func=_geometry_map)

    fc = sub.add_parser("fit-condition", help="fit + characterize the harmful-input detector (C4)")
    fc.add_argument("--config", required=True)
    fc.add_argument("--benign", default="orbench", help="benign benchmark for the negatives")
    fc.add_argument("--xstest-limit", type=int, default=200,
                    help="XSTest prompts for the over-refusal FPR (Item 4)")
    fc.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    fc.set_defaults(func=_fit_condition)

    vd = sub.add_parser("validate-drefuse",
                        help="construct-validity checks for d_refuse: ablation, template, "
                             "natural-refusal, naive-DIM (Item 2; run on an aligned model)")
    vd.add_argument("--config", required=True)
    vd.add_argument("--benign", default="orbench", help="harmless set for the naive-DIM contrast")
    vd.add_argument("--layers", type=int, nargs="*", default=None)
    vd.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    vd.set_defaults(func=_validate_drefuse)

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
    ev.add_argument("--force-op", default=None, choices=["raw_add", "project"],
                    help="force the wrapper's operator on all band layers (operator x geometry "
                         "conditions for the crossover interaction, Item 4)")
    ev.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--seeds", type=int, nargs="*", default=None)
    ev.add_argument("--hf-judge", action="store_true")
    ev.add_argument("--force-alpha", action="store_true",
                    help="silence the pre-registered-alpha mismatch warning (Item 3)")
    ev.set_defaults(func=_eval)

    ab = sub.add_parser("ablate", help="sweep one wrapper ablation axis (C4 ablations)")
    ab.add_argument("--config", required=True)
    ab.add_argument("--benchmark", required=True)
    ab.add_argument("--axis", required=True, choices=["alpha", "layers", "condition"])
    ab.add_argument("--alpha", type=float, default=8.0, help="fixed alpha for layers/condition axes")
    ab.add_argument("--alphas", type=float, nargs="*", default=[2, 4, 8, 16])
    ab.add_argument("--layer-sets", nargs="*", default=None,
                    help='for --axis layers, e.g. "13,14" "13,14,15,16"')
    ab.add_argument("--select-over", default=None,
                    help="for --axis alpha: over-refusal benchmark; freezes the argmax of "
                         "(harmful-refusal − over-refusal) to cache/alpha/<model>.json (Item 3)")
    ab.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    ab.add_argument("--limit", type=int, default=None)
    ab.add_argument("--seeds", type=int, nargs="*", default=None)
    ab.add_argument("--hf-judge", action="store_true")
    ab.set_defaults(func=_ablate)

    at = sub.add_parser("attack", help="adversarial robustness suite vs a defense (C5, Item 6)")
    at.add_argument("--config", required=True)
    at.add_argument("--defense", default="none",
                    choices=["none", "system_prompt", "abliteration", "cast", "wrapper"],
                    help="defended generator to attack (default: undefended model)")
    at.add_argument("--attack", default="gcg",
                    choices=["gcg", "gcg-adaptive", "gcg-detector", "pair", "multiturn"],
                    help="gcg=static; gcg-adaptive=through the defense; gcg-detector=evade the "
                         "condition; multiturn=cheap persona demo")
    at.add_argument("--behaviors", default="advbench",
                    help="behaviour benchmark ('advbench' uses the pre-registered eval split)")
    at.add_argument("--budgets", type=int, nargs="*", default=None,
                    help="query-budget grid for the ASR curve (default: config attack.budgets)")
    at.add_argument("--alpha", type=float, default=8.0, help="steering strength (steered defenses)")
    at.add_argument("--force-op", default=None, choices=["raw_add", "project"],
                    help="force the wrapper's operator on all band layers")
    at.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    at.add_argument("--limit", type=int, default=None, help="cap number of behaviours")
    at.add_argument("--seed", type=int, default=0)
    at.add_argument("--n-steps", type=int, default=None, help="override GCG steps")
    at.add_argument("--search-width", type=int, default=None, help="override GCG candidates/step")
    at.add_argument("--topk", type=int, default=None, help="override GCG top-k per position")
    at.add_argument("--penalty-lambda", type=float, default=None,
                    help="detector-aware hinge weight lambda (gcg-detector)")
    at.add_argument("--penalty-margin", type=float, default=None,
                    help="detector-aware hinge margin (gcg-detector)")
    at.set_defaults(func=_attack)

    rp = sub.add_parser("report", help="regenerate all tables + figures from runs.sqlite (M5)")
    rp.add_argument("--config", required=True)
    rp.add_argument("--out", default="report")
    rp.add_argument("--judge", default="rubric")
    rp.add_argument("--temperature", type=float, default=0.0)
    rp.set_defaults(func=_report)

    so = sub.add_parser("score", help="dual-score a responses parquet")
    so.add_argument("--responses", required=True)
    so.add_argument("--hf-judge", action="store_true", help="also run the HF classifier (#2)")
    so.add_argument("--out", default=None, help="write scored parquet here")
    so.set_defaults(func=_score)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
