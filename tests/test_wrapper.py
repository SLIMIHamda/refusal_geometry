import numpy as np

from asw.wrapper.condition import ConditionVector
from asw.wrapper.steer import branch_for_label, op_project_amplify, op_raw_add
from asw.wrapper.wrapper import Wrapper


def test_from_geometry_map_uses_measured_geometry():
    d = {13: np.zeros(4), 14: np.zeros(4)}
    amap = {13: {"label": "anti-aligned"}, 14: {"label": "aligned"}}
    w = Wrapper.from_geometry_map(None, None, d, amap, 8.0)
    assert w.branch_by_layer == {13: "raw_add", 14: "project"}


def test_from_geometry_map_force_op_overrides():
    d = {13: np.zeros(4), 14: np.zeros(4)}
    amap = {13: {"label": "anti-aligned"}, 14: {"label": "aligned"}}
    w = Wrapper.from_geometry_map(None, None, d, amap, 8.0, force_op="project")
    assert w.branch_by_layer == {13: "project", 14: "project"}
    w2 = Wrapper.from_geometry_map(None, None, d, amap, 8.0, force_op="raw_add")
    assert w2.branch_by_layer == {13: "raw_add", 14: "raw_add"}


def test_branch_for_label():
    assert branch_for_label("anti-aligned") == "raw_add"
    assert branch_for_label("aligned") == "project"
    assert branch_for_label("neutral") == "project"


def test_op_raw_add_uses_unit_direction():
    assert np.allclose(op_raw_add(np.zeros(4), np.array([2.0, 0, 0, 0]), 3.0), [3, 0, 0, 0])


def test_project_amplify_strengthens_aligned_geometry():
    d = np.array([1.0, 0, 0, 0])
    h = np.array([2.0, 1, 0, 0])              # positive component along d (aligned)
    assert op_project_amplify(h, d, 1.0)[0] > h[0]


def test_project_amplify_fails_on_anti_aligned():
    # the C1 falsification: amplifying the OWN component on negative geometry reduces refusal
    d = np.array([1.0, 0, 0, 0])
    h = np.array([-2.0, 1, 0, 0])             # negative component (anti-aligned)
    assert op_project_amplify(h, d, 1.0)[0] < h[0]


def test_project_amplify_batched():
    d = np.array([1.0, 0])
    out = op_project_amplify(np.array([[3.0, 0], [-3.0, 0]]), d, 1.0)
    assert out[0, 0] > 3 and out[1, 0] < -3


def test_condition_vector_separates_classes():
    rng = np.random.default_rng(0)
    harmful = rng.normal(0, 0.1, (50, 4)) + np.array([3, 0, 0, 0])
    benign = rng.normal(0, 0.1, (50, 4)) + np.array([-3, 0, 0, 0])
    cv = ConditionVector.fit(harmful, benign)
    preds = cv.predict(np.vstack([[3.0, 0, 0, 0], [-3.0, 0, 0, 0]]))
    assert bool(preds[0]) and not bool(preds[1])


def test_condition_save_load(tmp_path):
    cv = ConditionVector(np.array([1.0, 0, 0]), 0.5)
    p = tmp_path / "cond.npz"
    cv.save(p)
    cv2 = ConditionVector.load(p)
    assert np.allclose(cv2.direction, [1, 0, 0]) and cv2.threshold == 0.5
