# Runbook — Dolphin experiments end-to-end (`notebooks/asw_experiments.ipynb`)

A start-to-finish walkthrough of the `refusal_geometry` pipeline on the reference model
`cognitivecomputations/dolphin-2.9-llama3-8b`. One *Run All* takes you from a bare repo to a
report containing the C1, C2, C4, and C5 results.

## The approach in one picture

The whole paper is one pipeline; each notebook stage is one tested `asw` subcommand, and **every
number in the report traces back to `runs.sqlite`** (nothing is hand-transcribed):

```
download → extract d_refuse (C2) → anti-alignment map (C1) → fit condition (C4)
         → evaluate defenses → ablation(α) → adversarial (C5) → REPORT.md
                                    ↑ all write to runs.sqlite / cache/ ↑
```

- **C2 (`d_refuse`)** — the refusal *direction*, via same-prompt native-vs-forced-refusal
  difference-in-means (cancels topic/length confounds).
- **C1 (anti-alignment)** — project activations onto `d_refuse`; **`⟨y,d_refuse⟩ < 0` on an
  uncensored model is the headline** (the property naive steering assumes away). Dolphin should
  read anti-aligned at the band.
- **C4 (the wrapper)** — geometry-aware, condition-gated steering: fire only on harmful inputs
  (condition vector), and pick the operator per layer by the measured geometry sign.
- **C5 (robustness)** — attack every defense with identical code; the wrapper's **ASR-vs-budget
  curve vs undefended** is the result.

## Where to run it

Kaggle GPU notebook, T4 ×1. Dolphin-8B in bf16 fits a single T4 (`QUANT=None`); if you OOM, set
`QUANT="int8"`. The repo clones itself (needs the `GITHUB_TOKEN` Kaggle secret). Artifacts live
under `/kaggle/working/refusal_geometry/{cache,data,results,report}` and survive *Save Version*.

## Cell-by-cell walkthrough

| Stage (header) | What it does | Produces / how to read it |
|---|---|---|
| **1 · Clone** | Clones on first run, `git pull --ff-only` after. | `cwd` + last commit printed. |
| **2 · Environment** | Pins `transformers==4.44.2` etc., then runs `pytest -q`. | **Spine must be green** before continuing. |
| **3 · Control Panel** | All settings. For Dolphin the defaults are correct: `MODEL_CONFIG=configs/models/dolphin-llama3-8b.yaml`, `ALPHA=8.0`, `STEER_LAYERS=None` (→ config's `[13,14,15,16]`). Leave `RUN_ATTACKS=False` for the first pass. | Prints the resolved settings. |
| **4 · Helpers** | Defines `asw(...)` (runs a CLI subcommand with live logs), `cfg()`, cache `*_exists()` checks, `show_runs()`. | Prints whether caches already exist. |
| **5 · Download** | Fetches `harmbench, xstest, orbench, advbench` → `data/*.jsonl`. | Idempotent; skips if present. |
| **6 · Extract `d_refuse` (C2)** | `asw extract` over the AdvBench extract split `[0,200]` at layers 13–16. | `cache/drefuse/dolphin*.npz`; layer-consistency cosines logged (should be high/coherent). |
| **7 · Anti-alignment map (C1)** | `asw geometry-map` (centered projection vs the `orbench` neutral mean, Item 1). | `cache/geometry/*.json` + inline table & heatmap. **Look for `anti-aligned` labels / negative projection at 13–16** — the Dolphin C1 claim. |
| **8 · Fit condition (C4)** | `asw fit-condition` (harmful=AdvBench extract vs benign=`orbench`) at the mid-band layer. | `cache/condition/*.npz`; train separation accuracy logged (want ≫ 0.5). |
| **9 · Evaluate defenses** | `asw eval` for each of `none, system_prompt, abliteration, cast, wrapper` × `{harmbench, xstest}` × `SEEDS`. Resumes at prompt granularity. | Rows in `runs.sqlite` + pooled refusal-rate table. **Story:** wrapper ↑ refusal on harmbench while keeping xstest (over-refusal) low; `abliteration`/`cast` show why naive re-addition is worse. |
| **10 · Ablation (α)** | `asw ablate --axis alpha` on harmbench. | Alpha-tradeoff table + curve. |
| **11 · Adversarial (C5)** | *(only if `RUN_ATTACKS=True`)* runs `asw attack` for **static** (`none`, `wrapper`), **through-defense** (`gcg-adaptive`), **detector-aware** (`gcg-detector`). | Rows in `runs.sqlite` + inline `table_attacks` and the ASR-vs-budget figure. |
| **12 · Report** | `asw report` reads `runs.sqlite`. | `report/REPORT.md` + `report/tables/*.csv` + `report/figures/*.png`, rendered inline — **your global snapshot: C1 map, refusal-by-defense, ablations, C5 ASR-vs-budget, operator×geometry interaction.** |
| **13 · Manifest** | `show_runs(20)` | Provenance: every run with config hash, seed, status, env. |

## First pass (no attacks) — the fast path

1. Stages 1–4 (clone, env, panel, helpers). Confirm the spine tests pass.
2. *Run* stages 5→12 with `RUN_ATTACKS=False`. Full **C1/C2/C4** story + a complete report in
   ~tens of minutes on a T4.
3. Read stage 12's report: confirm Dolphin is **anti-aligned** (C1) and the **wrapper** raises
   harmful-refusal without wrecking XSTest.

## Second pass — turn on C5 (Tier-1)

Attacks are the expensive part (GCG optimizes a suffix per behavior). **Before trusting any
absolute ASR, validate the optimizer** — run once on the host:

```bash
python scripts/validate_gcg.py --model cognitivecomputations/dolphin-2.9-llama3-8b --n-behaviors 10
```

It reports baseline vs `run_gcg` ASR (and compares to nanoGCG if installed). Only once `run_gcg`
clearly beats the no-suffix baseline should you report absolute numbers; **until then report only
*relative* ASR across defenses** (all attacked by identical code).

Then set `RUN_ATTACKS=True` (keep `ATTACK_LIMIT=3`, `ATTACK_STEPS=40` for a demo) and *Run* stages
11→12. The report's **"Adversarial robustness — ASR vs query budget (C5)"** section populates. For
the paper, drop `ATTACK_LIMIT`/`--n-steps` and let the pre-registered config (`configs/base.yaml →
attack:`, 250 steps × 256 width = 64k queries, budgets `[8k,16k,32k,64k]`) drive it — see
[`docs/PRE_REGISTRATION_C5.md`](PRE_REGISTRATION_C5.md).

The same suite from a terminal:

```bash
for d in none system_prompt abliteration cast wrapper; do
  python -m asw.harness.cli attack --config configs/models/dolphin-llama3-8b.yaml --defense $d --attack gcg
done
python -m asw.harness.cli attack --config configs/models/dolphin-llama3-8b.yaml --defense wrapper --attack gcg-adaptive
python -m asw.harness.cli attack --config configs/models/dolphin-llama3-8b.yaml --defense wrapper --attack gcg-detector
python -m asw.harness.cli report --config configs/models/dolphin-llama3-8b.yaml
```

## Two things to internalize about the honest metric

- **Relative, not absolute** until `validate_gcg.py` passes.
- **Honest number = adaptive number**: the defense claim cites `gcg-adaptive` (through the steering
  hooks) / `gcg-detector` (evading the condition), *not* a static-suffix replay. The notebook runs
  exactly those.

## Aside — construct validity (Item 2) is *not* in the Dolphin flow, by design

`asw validate-drefuse` (ablate `d_refuse` → refusal must collapse) is only meaningful on an
**aligned** model, so run it against `configs/models/llama3-8b-instruct.yaml`, not Dolphin. It
surfaces in the report's "d_refuse construct validity" section when present.
