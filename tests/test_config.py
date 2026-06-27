from pathlib import Path

from asw.config import config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_base_composition():
    cfg = load_config(ROOT / "configs/models/dolphin-llama3-8b.yaml")
    assert cfg["model"]["id"].endswith("dolphin-2.9-llama3-8b")
    assert cfg["seeds"] == [0, 1, 2]                      # inherited from base.yaml
    assert cfg["model"]["steer_layers"] == [13, 14, 15, 16]
    assert cfg["splits"]["advbench"]["extract"] == [0, 200]


def test_hash_is_order_invariant_and_short():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 12


def test_hash_changes_on_meaningful_edit():
    base = load_config(ROOT / "configs/models/dolphin-llama3-8b.yaml")
    edited = load_config(ROOT / "configs/models/dolphin-llama3-8b.yaml")
    edited["model"]["steer_layers"] = [13, 14]
    assert config_hash(base) != config_hash(edited)
