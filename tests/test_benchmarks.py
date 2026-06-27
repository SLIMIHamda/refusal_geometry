from pathlib import Path

import pytest

from asw.data import load_benchmark

FIX = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_load_jsonl_task_and_fields():
    b = load_benchmark("harmbench", data_dir=FIX)
    assert b.task == "harmful" and len(b) == 5
    assert all(e.prompt for e in b.examples)
    assert b.examples[0].id == "hb_0" and b.examples[0].category == "explosives"


def test_limit():
    assert len(load_benchmark("harmbench", data_dir=FIX, limit=2)) == 2


def test_over_refusal_task():
    assert load_benchmark("xstest", data_dir=FIX).task == "over_refusal"


def test_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_benchmark("does_not_exist", data_dir=FIX)
