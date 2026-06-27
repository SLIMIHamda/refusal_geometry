import pandas as pd

from asw.data.benchmarks import Example
from asw.harness.generate import ScriptedGenerator, run_generation


def _exs(n):
    return [Example(id=i, prompt=f"p{i}") for i in range(n)]


def test_basic_generation():
    g = ScriptedGenerator(lambda p: f"resp:{p}")
    rows = run_generation(g, _exs(3), temperature=0.0, max_new_tokens=8, seed=0)
    assert len(rows) == 3
    assert rows[0]["response"] == "resp:p0"
    assert all(not r["reused"] for r in rows)


def test_resume_reuses_finished_prompts(tmp_path):
    g = ScriptedGenerator(lambda p: f"resp:{p}")
    rows = run_generation(g, _exs(3), temperature=0.0, max_new_tokens=8, seed=0)
    path = tmp_path / "r.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    # a different generator would emit "NEW"; resumed prompts must keep their old text
    g2 = ScriptedGenerator(lambda p: "NEW")
    rows2 = run_generation(
        g2, _exs(4), temperature=0.0, max_new_tokens=8, seed=0, resume_path=path
    )
    by = {r["prompt_id"]: r for r in rows2}
    assert by[0]["response"] == "resp:p0" and by[0]["reused"]
    assert by[3]["response"] == "NEW" and not by[3]["reused"]
