import glob

import pandas as pd

from asw import db as dbm
from asw.data.benchmarks import Benchmark, Example
from asw.harness.evaluate import evaluate_benchmark
from asw.harness.generate import ScriptedGenerator
from asw.scorers.judge import RubricJudge


class StubJudge:
    """Disagrees with the rubric on odd-indexed prompts, so agreement is non-trivial
    and Cohen's kappa is well-defined (both labels present)."""

    name = "stub"

    def label(self, prompt, response):
        idx = int(prompt.rsplit(" ", 1)[1])
        return "refusal" if idx % 2 == 0 else "comply"


def test_evaluate_end_to_end(tmp_path):
    con = dbm.connect(tmp_path / "runs.sqlite")
    bench = Benchmark(
        "harmbench", "harmful",
        [Example(id=i, prompt=f"do something harmful {i}") for i in range(4)],
    )
    gen = ScriptedGenerator(lambda p: "I'm sorry, but I cannot help with that.")
    judges = {"rubric": RubricJudge(), "stub": StubJudge()}
    cfg = {"decoding": {"temperatures": [0.0], "max_new_tokens": 8}}

    metrics = evaluate_benchmark(
        con, generator=gen, benchmark=bench, model_id="m", config=cfg, seed=0,
        judges=judges, decoding=cfg["decoding"], results_dir=tmp_path / "results",
    )

    key = "refusal_rate.rubric.T0.0"
    assert metrics[key]["rate"] == 1.0 and metrics[key]["n"] == 4
    # rubric = all refusal; stub disagrees on the 2 odd prompts -> 0.5 agreement
    assert metrics["agreement.T0.0"]["raw_agreement"] == 0.5
    assert not (metrics["agreement.T0.0"]["cohen_kappa"] != metrics["agreement.T0.0"]["cohen_kappa"])  # not NaN

    row = con.execute("SELECT status, metrics_json FROM runs").fetchone()
    assert row["status"] == "completed" and "refusal_rate" in row["metrics_json"]

    files = glob.glob(str(tmp_path / "results" / "eval_harmbench" / "*.parquet"))
    assert files and len(pd.read_parquet(files[0])) == 4


def _run(tmp_path, response, n=4):
    con = dbm.connect(tmp_path / "runs.sqlite")
    bench = Benchmark("harmbench", "harmful",
                      [Example(id=i, prompt=f"do something harmful {i}") for i in range(n)])
    cfg = {"decoding": {"temperatures": [0.0], "max_new_tokens": 8}}
    return evaluate_benchmark(
        con, generator=ScriptedGenerator(lambda p: response), benchmark=bench, model_id="m",
        config=cfg, seed=0, judges={"rubric": RubricJudge()}, decoding=cfg["decoding"],
        results_dir=tmp_path / "results")


def test_gibberish_is_no_longer_scored_as_compliance(tmp_path):
    """The dolphin POC defect, as a regression test.

    Steering at alpha=8 made every defense emit `inuinuinu...`; the rubric found no refusal
    marker, scored it `comply`, and the headline table read 0.00 refusal — indistinguishable
    from a model that simply complied. The refusal rate still reads 0.00 (the rubric is
    unchanged and that is correct), but the run is now self-evidently broken."""
    m = _run(tmp_path, "inuinu" * 30)

    assert m["refusal_rate.rubric.T0.0"]["rate"] == 0.0        # the misleading legacy number
    assert m["fluency.T0.0"]["degenerate_rate"] == 1.0         # ...now impossible to misread
    # refusal is UNMEASURABLE on a wholly degenerate run, not zero: the fluent denominator is 0
    assert m["refusal_rate_fluent.rubric.T0.0"]["n"] == 0
    assert m["refusal_rate_fluent.rubric.T0.0"]["rate"] != m["refusal_rate_fluent.rubric.T0.0"]["rate"]


def test_genuine_compliance_is_not_flagged(tmp_path):
    """The other half of the discrimination: a real compliant answer must stay fluent, or the
    fluency channel would simply relabel every low-refusal run as broken."""
    m = _run(tmp_path, "Step 1: gather the parts from a hardware store. Step 2: assemble them.")

    assert m["refusal_rate.rubric.T0.0"]["rate"] == 0.0
    assert m["fluency.T0.0"]["degenerate_rate"] == 0.0
    assert m["refusal_rate_fluent.rubric.T0.0"]["rate"] == 0.0
    assert m["refusal_rate_fluent.rubric.T0.0"]["n"] == 4


def test_degeneracy_flags_are_persisted_per_prompt(tmp_path):
    """The flag and its trigger reach the parquet, so a run can be re-audited without rerunning."""
    _run(tmp_path, "inuinu" * 30)
    pq = pd.read_parquet(glob.glob(str(tmp_path / "results" / "eval_harmbench" / "*.parquet"))[0])
    assert pq["degenerate"].tolist() == [1, 1, 1, 1]
    assert all(r.startswith("char_loop:") for r in pq["fluency_rule"])
