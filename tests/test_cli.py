import pytest

from asw.harness.cli import (
    _ablation_points, _build_generator, _condition_layer, _drefuse_path,
)


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


def test_ablation_points_alpha():
    pts = _ablation_points("alpha", alpha=8.0, alphas=[2, 8], layer_sets=[])
    assert [lbl for lbl, _ in pts] == ["alpha=2", "alpha=8"]
    assert pts[1][1] == {"alpha": 8}


def test_ablation_points_condition():
    pts = _ablation_points("condition", alpha=8.0, alphas=[], layer_sets=[])
    assert [lbl for lbl, _ in pts] == ["cond=on", "cond=off"]
    assert pts[0][1] == {"alpha": 8.0, "use_condition": True}
    assert pts[1][1]["use_condition"] is False


def test_ablation_points_layers():
    pts = _ablation_points("layers", alpha=8.0, alphas=[], layer_sets=[[13, 14], [13, 14, 15, 16]])
    assert pts[0][0] == "layers=13,14"
    assert pts[1][1] == {"alpha": 8.0, "layers": [13, 14, 15, 16]}


def test_ablation_points_bad_axis():
    with pytest.raises(SystemExit):
        _ablation_points("nonsense", alpha=8.0, alphas=[], layer_sets=[])


def test_attack_cli_parses_and_dispatches(monkeypatch):
    """The `attack` subcommand wires its flags to _attack (dispatch is stubbed so no model loads)."""
    import asw.harness.cli as cli

    captured = {}

    def fake(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_attack", fake)
    rc = cli.main(["attack", "--config", "c.yaml", "--attack", "gcg-detector",
                   "--defense", "wrapper", "--behaviors", "advbench",
                   "--budgets", "500", "1000", "--penalty-lambda", "0.5", "--limit", "5"])
    a = captured["args"]
    assert rc == 0 and a.attack == "gcg-detector" and a.defense == "wrapper"
    assert a.budgets == [500, 1000] and a.penalty_lambda == 0.5 and a.limit == 5


def test_attack_cli_rejects_unknown_attack():
    import asw.harness.cli as cli

    with pytest.raises(SystemExit):        # argparse rejects a bad --attack choice
        cli.main(["attack", "--config", "c.yaml", "--attack", "nope"])
