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
