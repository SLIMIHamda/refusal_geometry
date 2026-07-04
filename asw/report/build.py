"""Orchestrate the report: tables -> CSV, figures -> PNG, all stitched into REPORT.md.

Robust to a partial manifest — each section is skipped (with a note) when its runs are
absent, so the same command works at M1 (geometry only) and at M5 (everything)."""
from __future__ import annotations

from pathlib import Path

from . import figures, tables
from .load import load_runs


def _md_table(df) -> str:
    """GitHub pipe table if `tabulate` is present, else a fenced plain-text table."""
    if df.empty:
        return "_(no runs yet)_"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def build_report(db_path, out_dir, *, judge: str = "rubric", temperature=0.0) -> Path:
    out = Path(out_dir)
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(db_path)
    n_runs = 0 if runs.empty else len(runs)
    md = ["# Experiment report",
          f"_Generated from `{db_path}` — {n_runs} runs, scorer=`{judge}`, T={temperature}._", ""]

    def section(title, df, csv_name, fig=None):
        md.append(f"## {title}\n")
        if df.empty:
            md.append("_(no runs yet)_\n")
            return
        df.to_csv(tdir / csv_name, index=False)
        md.append(_md_table(df) + "\n")
        if fig is not None:
            md.append(f"![{title}]({fig.relative_to(out).as_posix()})\n")

    # C1 — anti-alignment map
    geo = tables.table_geometry(runs)
    geo_fig = figures.fig_anti_alignment_map(geo, fdir / "anti_alignment_map.png")
    section("Anti-alignment map (C1)", geo, "geometry.csv", geo_fig)

    # d_refuse construct validity (Item 2)
    section("d_refuse construct validity (Item 2)", tables.table_validation(runs), "validation.csv")

    # detector characterization (Item 4): held-out AUC / TPR / FPR + XSTest FPR + threshold sweep
    detector = tables.table_detector(runs)
    sweep, tau = tables.latest_threshold_sweep(runs)
    det_fig = figures.fig_threshold_sweep(sweep, fdir / "detector_threshold.png", tau=tau) \
        if not detector.empty else None
    section("Detector characterization (Item 4)", detector, "detector.csv", det_fig)

    # main results — refusal rate by defense (prompt-clustered CI from the per-run parquet)
    refusal = tables.table_refusal(runs, judge=judge, temperature=temperature,
                                   results_dir=Path(db_path).parent)
    refusal_fig = None
    if not refusal.empty:
        bench = sorted(refusal["benchmark"].dropna().unique())[0]
        refusal_fig = figures.fig_refusal_bars(refusal, bench, fdir / "refusal_by_defense.png")
    section("Refusal rate by defense (main results)", refusal, "refusal.csv", refusal_fig)

    # ablations
    for axis in ("alpha", "layers", "condition", "neutral_op"):
        at = tables.table_ablation(runs, axis, judge=judge, temperature=temperature)
        fig = figures.fig_alpha_tradeoff(at, fdir / "alpha_tradeoff.png") if axis == "alpha" else None
        section(f"Ablation: {axis}", at, f"ablation_{axis}.csv", fig)

    # adversarial robustness — ASR vs query budget (C5, Item 6)
    attacks = tables.table_attacks(runs)
    asr_fig = figures.fig_asr_vs_budget(attacks, fdir / "asr_vs_budget.png") \
        if not attacks.empty else None
    section("Adversarial robustness — ASR vs query budget (C5)", attacks, "attacks.csv", asr_fig)

    # scorer adjudication (Item 7): dump the dual-scorer disagreement set; if a human-labeled file
    # is present next to the DB, report which scorer the human sides with.
    md.append("## Scorer adjudication (Item 7)\n")
    try:
        dis = tables.disagreements(runs, Path(db_path).parent, temperature=temperature)
    except Exception as e:  # noqa: BLE001 - never break the report on a bad parquet
        dis = None
        md.append(f"_not computed: {type(e).__name__}: {e}_\n")
    if dis is None:
        pass
    elif dis.empty:
        md.append("_(no dual-scorer disagreements — run `eval --hf-judge` so both scorers label "
                  "the same responses)_\n")
    else:
        dis.to_csv(tdir / "adjudication_disagreements.csv", index=False)
        labeled = Path(db_path).parent / "adjudication_labeled.csv"
        md.append(f"_{len(dis)} responses where `rubric` and `hf_classifier` disagree → "
                  f"`tables/adjudication_disagreements.csv`. Hand-label a sample (add a "
                  f"`human_label` column of refusal|comply), save as `{labeled.as_posix()}`, and "
                  f"re-run the report for the verdict._\n")
        if labeled.exists():
            import pandas as pd
            summ = tables.adjudication_summary(pd.read_csv(labeled))
            if summ:
                md.append(f"_Human adjudication (n={summ['n']}): rubric agrees "
                          f"{summ['agree_rubric']:.2f}, hf_classifier agrees "
                          f"{summ['agree_hf_classifier']:.2f} → **{summ['winner']}** sides with "
                          f"the human more often._\n")

    # operator x geometry crossover interaction (Item 4 headline statistic)
    md.append("## Operator x geometry interaction (Item 4)\n")
    try:
        res = tables.crossover_from_runs(runs, Path(db_path).parent, judge=judge,
                                         temperature=temperature)
    except Exception as e:  # statsmodels absent, GEE non-convergence, etc. — never break the report
        res = None
        md.append(f"_not computed: {type(e).__name__}: {e}_\n")
    if res is None:
        md.append("_(no operator-forced wrapper runs yet — run `eval --defense wrapper "
                  "--force-op {raw_add,project}` on aligned and anti-aligned models)_\n")
    elif res.get("insufficient"):
        md.append(f"_insufficient design: operators={res['operators']}, geometry={res['geometry']}, "
                  f"n_obs={res['n_obs']} — need 'none' + a forced operator across both geometry "
                  "classes._\n")
    else:
        import pandas as pd
        idf = pd.DataFrame([{"term": k, **v} for k, v in res["interactions"].items()])
        if not idf.empty:
            idf.to_csv(tdir / "interaction.csv", index=False)
            md.append(_md_table(idf) + "\n")
        md.append(f"_GEE logistic (prompt-clustered); n_obs={res['n_obs']}, "
                  f"n_prompts={res['n_groups']}, operators={res['operators']}, "
                  f"geometry={res['geometry']}._\n")

    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    return out
