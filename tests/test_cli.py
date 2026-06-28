import pytest

from asw.harness.cli import _build_generator, _condition_layer, _drefuse_path


def _cfg(tmp):
    return {"paths": {"cache_dir": str(tmp)},
            "model": {"id": "x/y", "steer_layers": [13, 14, 15, 16]}}


def test_build_generator_none(tmp_path):
    g = _build_generator(_cfg(tmp_path), None, None, "none", 8.0)
    assert g.__class__.__name__ == "HFGenerator" and g.system_prompt is None


def test_build_generator_system_prompt(tmp_path):
    g = _build_generator(_cfg(tmp_path), None, None, "system_prompt", 8.0)
    assert g.system_prompt is not None


def test_build_generator_requires_drefuse(tmp_path):
    with pytest.raises(SystemExit):
        _build_generator(_cfg(tmp_path), None, None, "abliteration", 8.0)


def test_build_generator_cast_requires_condition(tmp_path):
    # provide a d_refuse cache so it advances to the condition check
    import numpy as np

    from asw.geometry.extract import save_drefuse
    save_drefuse(_drefuse_path(_cfg(tmp_path)), {13: np.zeros(4)})
    with pytest.raises(SystemExit):
        _build_generator(_cfg(tmp_path), None, None, "cast", 8.0)


def test_condition_layer_default():
    assert _condition_layer({"model": {"steer_layers": [13, 14, 15, 16]}}) == 15
