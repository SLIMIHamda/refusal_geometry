# Refusal Geometry & Activation-Steering Safety Wrapper (`asw`)

Journal extension (Elsevier KBS/ESWA) of the activation-steering refusal thesis:
characterize the **anti-alignment** geometry of a zoo of open-weight LLMs, localize the
refusal circuit causally, and build a **geometry-aware, input-conditional steering
wrapper** validated **under adversarial attack**.

Contributions C1–C5 and the 50-day plan (gates M1–M5) live in the three briefings at the
repo root. Every figure/table in Briefing 3 is the output of a specific workstream.

## Status: Step 1 — unified eval harness (M1)
GPU-free spine, fully tested on the authoring box (26 tests). Everything downstream
writes into it; the GPU pieces (model loader, steering hooks) hang off it in Steps 3+.

| Module | Role |
|---|---|
| `asw/repro.py`   | global seeding, deterministic flags, env capture |
| `asw/config.py`  | YAML registry, `_base_` composition, deterministic `config_hash` |
| `asw/db.py`      | SQLite run index (Tab A manifest) + Parquet per-prompt rows; resume |
| `asw/runlog.py`  | timed `run_context` that stamps every run row |
| `asw/data/benchmarks.py` | uniform loader (HarmBench/StrongREJECT/XSTest/… → `Benchmark`) |
| `asw/scorers/refusal.py` | string-rubric refusal scorer (#1) |
| `asw/scorers/judge.py`   | judge interface + offline RoBERTa classifier (#2) + LLM-judge |
| `asw/eval/metrics.py`    | Clopper-Pearson/Wilson CIs, bootstrap seed CIs, paired tests, agreement |
| `asw/harness/generate.py`| batched generation, resumable at prompt granularity |
| `asw/harness/evaluate.py`| generate → dual-score → metrics → persist, wrapped in `run_context` |
| `asw/harness/cli.py`     | `asw selfcheck`, `asw score` |

Benchmark loaders read `data/<name>.jsonl` (offline-first); tiny samples live in
`data/fixtures/` for tests. Real benchmark JSONLs are produced by a download script (Step 3).

## Quickstart
```bash
pip install -r requirements.txt          # GPU stack stays commented for authoring
pytest -q                                # spine tests
python -m asw.harness.cli selfcheck --config configs/models/dolphin-llama3-8b.yaml
```

## Reproducibility contract
- One SQLite file is the source of truth; every paper table is a `WHERE config_hash=…` query.
- `run_id = sha256(experiment|model_id|config_hash|seed)[:16]` → idempotent resume after
  Kaggle 12h cutoffs / RunPod spot reclaim.
- GPU hosts pin Python 3.10 + the thesis stack (`requirements.txt`); the spine also runs on
  3.12 locally. **Qwen-2.5 needs `transformers>=4.43`** — see the note in `requirements.txt`.
