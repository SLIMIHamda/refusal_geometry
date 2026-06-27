from pathlib import Path

import pytest

from asw.config import config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]
MODEL_CFGS = sorted((ROOT / "configs" / "models").glob("*.yaml"))
REQUIRED = ("id", "family", "alignment", "recipe", "dtype", "n_layers", "d_model", "steer_layers")


@pytest.mark.parametrize("path", MODEL_CFGS, ids=lambda p: p.stem)
def test_model_config_valid(path):
    cfg = load_config(path)
    m = cfg["model"]
    for k in REQUIRED:
        assert k in m, f"{path.name} missing model.{k}"
    assert m["alignment"] in ("aligned", "uncensored")
    assert all(0 <= layer < m["n_layers"] for layer in m["steer_layers"]), \
        f"{path.name}: steer_layers out of range for n_layers={m['n_layers']}"
    assert cfg["seeds"] == [0, 1, 2]            # base.yaml inherited
    assert len(config_hash(cfg)) == 12


def test_zoo_meets_briefing_requirements():
    cfgs = [load_config(p)["model"] for p in MODEL_CFGS]
    families = {c["family"] for c in cfgs}
    uncensored = [c for c in cfgs if c["alignment"] == "uncensored"]
    recipes = {c["recipe"] for c in uncensored}
    assert len(cfgs) >= 6                       # >=6 models
    assert len(families) >= 3                   # >=3 families
    assert {"synthetic-sft", "abliterated"} <= recipes   # both uncensored recipes
    assert len(uncensored) >= 3
