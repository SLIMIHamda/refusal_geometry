import pytest

from asw.data.download import SPECS, normalize_row, write_jsonl


def test_normalize_harmful_with_category():
    spec = {"prompt": "prompt", "category": "category"}
    row = normalize_row("harmbench", {"prompt": "do X", "category": "cyber"}, spec, 3)
    assert row == {"id": "harmbench_3", "prompt": "do X", "category": "cyber"}


def test_normalize_drops_missing_category():
    spec = {"prompt": "question", "answer": "answer"}
    row = normalize_row("gsm8k", {"question": "2+2?", "answer": "4"}, spec, 0)
    assert row == {"id": "gsm8k_0", "prompt": "2+2?", "answer": "4"}
    assert "category" not in row


def test_specs_cover_core_axes():
    for name in ("advbench", "harmbench", "strongreject", "xstest", "orbench"):
        assert name in SPECS and "prompt" in SPECS[name]


def test_write_jsonl_roundtrip(tmp_path):
    from asw.data.benchmarks import from_jsonl

    rows = [{"id": "a_0", "prompt": "hello"}, {"id": "a_1", "prompt": "world"}]
    path = write_jsonl(tmp_path, "advbench", rows)
    bench = from_jsonl("advbench", path)
    assert len(bench) == 2 and bench.examples[1].prompt == "world"
    assert bench.task == "harmful"
