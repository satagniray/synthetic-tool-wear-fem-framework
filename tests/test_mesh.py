import numpy as np
import pytest

from tool_wear_demo.core.mesh import Mesh, MeshSpec, ToolGeometry, create_tool_mesh, mesh_from_config
from tool_wear_demo.core.quality import MeshQualityChecker, signed_areas
from tool_wear_demo.io import csv_io
from tool_wear_demo.io.config import default_config


def test_mesh_size_matches_spec(small_mesh):
    spec = MeshSpec(n_flank=12, n_edge=6, n_rake=14, n_depth=10, surface_growth=1.15, depth_exponent=1.5)
    n_u = spec.n_contour_elements
    assert small_mesh.n_elements == n_u * spec.n_depth
    assert small_mesh.n_nodes == (n_u + 1) * (spec.n_depth + 1)


def test_all_elements_counter_clockwise_and_valid(small_mesh):
    assert np.all(signed_areas(small_mesh.coords, small_mesh.elements) > 0)
    assert MeshQualityChecker(0.05).check(small_mesh.coords, small_mesh.elements).valid


def test_default_mesh_is_valid():
    mesh = mesh_from_config(default_config())
    report = MeshQualityChecker(0.05).check(mesh.coords, mesh.elements)
    assert report.valid and report.min_quality > 0.3


def test_surface_tags(small_mesh):
    counts = {t: int((small_mesh.surface_type == t).sum()) for t in ("flank", "edge", "rake", "back", "interior")}
    assert counts["flank"] == 12 - 1
    assert counts["edge"] == 6 + 1
    assert counts["rake"] == 14 - 1
    assert counts["back"] > 0 and counts["interior"] > 0
    assert np.all(small_mesh.is_surface == (small_mesh.surface_type != "interior"))


def test_tool_area_is_plausible(small_mesh):
    g = ToolGeometry()
    assert 0.3 * g.width_mm * g.height_mm < small_mesh.area() < g.width_mm * g.height_mm


def test_edge_nodes_lie_on_the_hone_radius(small_mesh):
    g = ToolGeometry()
    edge = small_mesh.coords[small_mesh.surface_type == "edge"]
    # least-squares circle fit through the edge nodes
    A = np.column_stack([2 * edge, np.ones(len(edge))])
    b = (edge**2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    radius = np.sqrt(c + cx**2 + cy**2)
    assert radius == pytest.approx(g.edge_radius_mm, rel=1e-6)
    assert np.allclose(np.hypot(edge[:, 0] - cx, edge[:, 1] - cy), g.edge_radius_mm)


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        ToolGeometry(edge_radius_mm=0.0)
    with pytest.raises(ValueError):
        ToolGeometry(rake_angle_deg=45.0)
    with pytest.raises(ValueError):
        MeshSpec(n_edge=1)


def test_with_coords_keeps_topology(small_mesh):
    moved = small_mesh.with_coords(small_mesh.coords + 0.001)
    assert np.array_equal(moved.elements, small_mesh.elements)
    assert not np.shares_memory(moved.coords, small_mesh.coords)


def test_mesh_rejects_bad_connectivity():
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 2)), np.array([[0, 1, 2, 3]]), ["interior"] * 3)


def test_csv_round_trip(tmp_path, small_mesh):
    csv_io.write_nodes(tmp_path / "n.csv", small_mesh)
    csv_io.write_elements(tmp_path / "e.csv", small_mesh)
    back = csv_io.read_mesh(tmp_path / "n.csv", tmp_path / "e.csv")
    assert np.allclose(back.coords, small_mesh.coords)
    assert np.array_equal(back.elements, small_mesh.elements)
    assert np.array_equal(back.surface_type, small_mesh.surface_type)
