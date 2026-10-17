import numpy as np
import pytest

from tool_wear_demo.core.quality import MeshQualityChecker, corner_jacobians, element_metrics, signed_areas
from tool_wear_demo.workflow.wear_loop import build_cycle_runner

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
QUAD = np.array([[0, 1, 2, 3]])


def test_valid_quad_accepted():
    report = MeshQualityChecker(0.05).check(SQUARE, QUAD)
    assert report.valid
    assert report.min_area == pytest.approx(1.0)
    assert report.min_quality == pytest.approx(1.0)
    assert report.min_angle_deg == pytest.approx(90.0)
    assert report.max_aspect_ratio == pytest.approx(1.0)


def test_inverted_quad_rejected():
    flipped = SQUARE[[0, 3, 2, 1]]
    assert signed_areas(flipped, QUAD)[0] < 0
    assert not MeshQualityChecker().check(flipped, QUAD).valid


def test_bow_tie_rejected_even_with_positive_area_parts():
    bow = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    _, scaled = corner_jacobians(bow, QUAD)
    assert scaled.min() <= 0
    assert not MeshQualityChecker().check(bow, QUAD).valid


def test_non_convex_quad_rejected():
    dart = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.2], [0.0, 1.0]])
    assert signed_areas(dart, QUAD)[0] > 0  # positive area ...
    assert not MeshQualityChecker().check(dart, QUAD).valid  # ... but a folded corner


def test_threshold_on_scaled_jacobian():
    skew = np.array([[0.0, 0.0], [1.0, 0.0], [1.9, 0.1], [0.9, 0.1]])
    assert MeshQualityChecker(0.0).check(skew, QUAD).valid
    assert not MeshQualityChecker(0.5).check(skew, QUAD).valid


def test_metrics_shapes(small_mesh):
    m = element_metrics(small_mesh.coords, small_mesh.elements)
    for key in ("area", "min_scaled_jacobian", "min_angle_deg", "min_edge_length", "aspect_ratio"):
        assert m[key].shape == (small_mesh.n_elements,)


def test_report_serialises(small_mesh):
    d = MeshQualityChecker().check(small_mesh.coords, small_mesh.elements).to_dict()
    assert set(d) >= {"valid", "min_area", "min_quality", "retries"}
    assert isinstance(d["valid"], bool)


def test_retry_scaling_restores_valid_geometry(small_mesh, small_config):
    """A deliberately oversized wear step is reduced automatically until the mesh is valid."""
    config = dict(small_config)
    config["wear"] = dict(config["wear"], target_max_surface_displacement_um=400.0)
    config["quality"] = dict(config["quality"], max_retries=10)
    runner = build_cycle_runner(config, small_mesh)
    result = runner.run(small_mesh, 1)
    assert result.quality.valid
    assert result.quality.retries >= 1
    assert result.quality.displacement_scale == pytest.approx(0.5**result.quality.retries)
    assert result.max_surface_move_um < 400.0


def test_cycle_fails_cleanly_without_retries(small_mesh, small_config):
    from tool_wear_demo.workflow.cycle_runner import CycleFailedError

    config = dict(small_config)
    config["wear"] = dict(config["wear"], target_max_surface_displacement_um=400.0)
    config["quality"] = dict(config["quality"], max_retries=0)
    runner = build_cycle_runner(config, small_mesh)
    with pytest.raises(CycleFailedError):
        runner.run(small_mesh, 1)
