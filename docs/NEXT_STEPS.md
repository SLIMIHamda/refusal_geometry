# Next steps

State as of 12 August 2026. Results are in `docs/DOLPHIN_REFUSAL_RESULTS.pdf`.
Raw data: `results_POC_notebook/results_bundle (1)/` plus the D1-D7 read-out
(`diagnose_steering.json`, currently only in the docx the student sent).

## Done (12 August 2026)

- **Step 1 (scorer) — complete.** The 12 advisory markers are in `asw/scorers/refusal.py`
  with tests. Re-scoring the stored bundles reproduces the headline numbers exactly
  (baseline 1→**4%**, cast 18→**36%**, abliteration 22→**45%**), catches 71 real refusals
  across 1800 responses with zero false positives on manual audit. Both
  `results_POC_notebook/results_bundle/` and `.../results_bundle (1)/` are re-scored in place.
- **Step 3 (validate-drefuse) — notebook stage wired**, off by default. Still needs a GPU +
  gated HF access to run (see below).

Correction to the plan below: re-scoring is **not** done by `audit_generations.py` (that only
touches the fluency channel). Use the new **`scripts/rescore_runs.py`** — it re-applies the
rubric to the parquet `label_rubric` *and* backfills the `runs.sqlite` metrics, because the
report reads refusal from both (main table ← parquet, dose-response table ← DB metrics).

## Where we are

Dolphin-2.9-Llama3-8B refuses 4% of harmful prompts. Adding the refusal direction at
layers 13-16 with `alpha=1.0` raises this to 36% (gated) or 45% (ungated), with 14% /
29% over-refusal on XSTest. Output stays fluent. Refusal rises with alpha in a smooth
curve, so the intervention is controlling behaviour rather than breaking the model.

Both earlier bugs are fixed and verified:
- alpha was 8, which is ~98% of the hidden-state norm. Now 1.0, ~12%.
- `d_refuse` was measuring sentence structure. Cosine with a one-word placebo went from
  +0.60 to -0.17, which passes the pre-registered gate of 0.35.

## ~~Do first (no GPU)~~ — DONE

**1. Add the missing refusal markers to `asw/scorers/refusal.py`.** ✅ Done — see the markers
listed above, plus `tests/test_refusal.py`. Re-score any bundle with:

```
python scripts/rescore_runs.py --results-dir <bundle-dir>          # preview
python scripts/rescore_runs.py --results-dir <bundle-dir> --write  # apply (back up first)
```

*Correction to the earlier "fixes the alpha=2 point" claim — it conflated two different sweeps:*

- **wrapper ablation**, n=100, in `runs.sqlite` (stage 10). Inert (project-amplify on
  negative-projection layers). After re-scoring its harmbench points are flat at base rate
  (0.04 / 0.05 / 0.02 / 0.02 for alpha 0.25 / 0.5 / 1 / 2); at alpha=2 it is already **33%
  degenerate**, so the markers do not move it. The re-score does *not* recover this point.
- **D5/D7 raw-add ladder**, n=20, in `report/diagnose_steering.json` (the diagnostic). A real
  raw-add dose-response — **this is what Figure 2 of the PDF plots**. Confirmed from the read-out:
  refusal `0.05 → 0.10 → 0.35 → 0.00 → 0.00` (D5) with degen `0 → 0 → 0 → 0 → 0.85`. The alpha=2
  row reads 0.00 **at degen 0.00** — the model is fully fluent, not broken — because the refusal
  *style* shifts from direct ("I'm sorry, it's against the rules") at alpha=1 to redirect ("I
  strongly encourage you to seek professional help and prioritize safety") at alpha=2, and the old
  scorer's vocabulary didn't follow. So the true peak refusal may sit at alpha=2, entirely hidden.

The `seek professional help` / `seeking professional help` markers are now **added** (FP-tested: 1
correct flip / 1800 harness responses; `i strongly encourage`/`i strongly recommend` deliberately
excluded — compliance uses them). But `rescore_runs.py` still cannot fix the stored D5 point: the
diagnostic persists only per-alpha rates plus a 110-char sample, not the 20 raw responses, so
nothing on disk can be re-scored. It needs the **diagnostic re-run** (`scripts/diagnose_steering.py
--alphas 0.25 0.5 1 2 4`, GPU, ~14 min), which now picks up all the new markers automatically.

What genuinely does not exist yet is a **gated `cast` sweep at n=100** through the harness
(step 2). Freezing alpha waits on that.

## Do next (needs GPU)

**2. Re-run the alpha sweep on the operator that works.**

The existing ablation swept the `wrapper`, which uses project-amplify and is inert at
every alpha. So we have no alpha curve for raw-addition, which is the operator that
actually induces refusal. Sweep `cast` over `alpha in {0.5, 1, 2, 3}` after step 1.

**3. Run `validate-drefuse` on an aligned model.** (Stage now wired — just needs a GPU + access.)

This is the causal test: project the direction out of a model that does refuse and check
refusal collapses. The notebook stage exists now — **Stage 6b**, gated on
`RUN_VALIDATE_DREFUSE = False` in the Control Panel. When enabled it:
- prechecks gated HF access to `meta-llama/Meta-Llama-3-8B-Instruct` and skips (no GPU spent)
  if the `HF_TOKEN` secret is missing or access has not been granted,
- runs `asw extract` on `configs/models/llama3-8b-instruct.yaml` first (the `d_refuse` cache is
  keyed by model — the aligned run gets its own `.npz`, so it never clobbers Dolphin's),
- runs `asw validate-drefuse` and prints a PASS/FAIL table for the ablation and format-confound
  gates.

To run: set `RUN_VALIDATE_DREFUSE = True`, ensure the `HF_TOKEN` secret is set and access to the
gated repo is approved, and (on a single T4) set `QUANT = "int8"` so the 8B model fits.

**4. More seeds.**

Everything so far is seed 0, greedy, n=100. The gap between gated and ungated steering
is inside sampling error. Run seeds 0, 1, 2 before any number goes in the paper.

## Open question for the contribution

All four band layers came back labelled `neutral`, and neutral routes to project-amplify,
which does nothing here. If we change neutral to raw-addition, the geometry-aware wrapper
becomes byte-identical to `cast`: same detector, same layers, same operator everywhere.

So on this model the geometry router has nothing to route. Either find a model with mixed
layer geometry, or reframe C4 around the detector plus operator choice rather than
per-layer routing.

The `neutral_op` ablation (`asw ablate --axis neutral_op`) will confirm the mechanism, but
it will not resolve this.

## Practical notes

- The notebook is `notebooks/asw_experiments.ipynb` at commit `161ef09` or later. Kaggle
  does not auto-update it. Repo code does update, through cell 2's `git pull`.
- Stage 14 bundles everything and refuses to write a partial zip. Ignore any older
  bundling command in chat history.
- `cache/geometry/` goes stale whenever `d_refuse` changes. Delete both together.
- Deterministic: greedy decoding at seed 0 reproduced byte-identical output across two
  runs three days apart.

## Results hygiene & scorer_version

Two things update on Kaggle by *different* routes:

- **Library + scripts** (`asw/`, `scripts/`) update automatically via cell 2's `git pull`. So a
  fresh eval immediately scores with the new markers and stamps `scorer_version` — you do **not**
  need the new notebook for the scorer fix to take effect.
- **The notebook cells themselves** (Stage 6b, the `VALIDATE_CONFIG`/`RUN_VALIDATE_DREFUSE`
  toggles, the `hf_access_ok` helper) do **not** auto-update — Kaggle runs its own saved copy, and
  the pull only refreshes the on-disk file, not the running notebook. Re-import the merged
  `.ipynb` to get them.

**The pooling guard.** `asw report` now *fails loudly* if a pooled group mixes `scorer_version`s —
`(model, benchmark, defense)` for the main table, `(benchmark, point)` for ablation. It does not
object to old and new runs merely coexisting in `runs.sqlite`; only to averaging two scorers into
one number. Scope:

- A **complete re-run** overwrites each unit (there is no skip-if-done guard on `asw eval`), so
  `results/` ends up uniformly new-scorer — nothing to clean up.
- The guard trips on **orphaned old runs** — config drift under one defense, or a dropped seed —
  that share a group with new ones. This is the "old runs pool with new" hazard, now caught rather
  than silently averaged.

Keeping old and new apart (all config-path-driven, so fully within repo standards):

- **Retire old:** archive `results/` → `results_archive_YYYYMM/` before a fresh pass.
- **Keep both live:** point the pass at its own `paths.results_db` / `paths.results_dir`.
- **Unify + correct (preferred for the paper):** `python scripts/rescore_runs.py --results-dir
  results --write` — CPU-only, needs the parquets. Re-scores in place, backfills `scorer_version`
  on legacy rows via the additive migration, and stamps provenance. The old numbers were
  *undercounted* by the buggy scorer, so quarantining them keeps the wrong values; re-scoring fixes
  them and keeps a single manifest.

`extract` / `geometry-map` / `fit-condition` carry no refusal metric, so `scorer_version` stays
NULL for them and they never trip the guard.
