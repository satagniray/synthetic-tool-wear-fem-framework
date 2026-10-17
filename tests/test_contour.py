import numpy as np
import pytest

from tool_wear_demo.core.contour import (
    ToolContour,
    boundary_loop,
    distance_to_polyline,
    points_in_polygon,
    polygon_area,
    resample_polyline,
)


def test_boundary_loop_is_closed_and_ccw(small_mesh):
    loop = boundary_loop(small_mesh)
    assert len(set(loop.tolist())) == loop.size
    assert polygon_area(small_mesh.coords[loop]) > 0
    assert np.all(small_mesh.is_surface[loop])
    # consecutive boundary nodes share an element edge
    edges = {tuple(e) for e in small_mesh.unique_edges().tolist()}
    for a, b in zip(loop, np.roll(loop, -1)):
        assert (min(a, b), max(a, b)) in edges


def test_contour_order_flank_edge_rake(contour):
    types = contour.surface_type.tolist()
    blocks = list(dict.fromkeys(types))
    assert blocks == ["flank", "edge", "rake"]
    # contiguous blocks
    changes = sum(1 for a, b in zip(types, types[1:]) if a != b)
    assert changes == 2


def test_contour_is_spatially_ordered(small_mesh, contour):
    pts = contour.points(small_mesh.coords)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.all(seg > 0)
    assert seg.max() < 0.2  # neighbours really are neighbours
    assert pts[0, 0] > pts[-1, 0] or pts[0, 1] < pts[-1, 1]  # flank end low/right, rake end high


def test_inward_normals_point_into_tool(small_mesh, contour):
    n = contour.inward_normals(small_mesh.coords)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    probe = contour.points(small_mesh.coords) + 1e-4 * n
    assert points_in_polygon(probe, contour.boundary_polygon(small_mesh.coords)).all()
    outside = contour.points(small_mesh.coords) - 1e-4 * n
    assert not points_in_polygon(outside, contour.boundary_polygon(small_mesh.coords)).any()


def test_rake_and_flank_normals_match_geometry(small_mesh, contour):
    n = contour.inward_normals(small_mesh.coords)
    flank_n = n[contour.mask("flank")]
    rake_n = n[contour.mask("rake")]
    assert np.all(flank_n[:, 1] > 0.9)  # flank: material is above
    assert np.all(rake_n[:, 0] > 0.9)  # rake: material is to the right


def test_signed_edge_distance(small_mesh, contour):
    d = contour.signed_edge_distance(small_mesh.coords)
    assert np.all(d[contour.mask("flank")] < 0)
    assert np.all(d[contour.mask("rake")] > 0)
    frac = contour.edge_fraction(small_mesh.coords)
    edge = frac[contour.mask("edge")]
    assert edge[0] == pytest.approx(0.0) and edge[-1] == pytest.approx(1.0)
    assert np.all(np.isnan(frac[~contour.mask("edge")]))


def test_anchors_are_fixed_back_nodes(small_mesh, contour):
    a0, a1 = contour.anchor_index
    assert small_mesh.surface_type[a0] == "back" and small_mesh.surface_type[a1] == "back"
    fr = contour.anchored_fractions(small_mesh.coords)
    assert np.all(np.diff(fr) > 0) and 0 < fr[0] and fr[-1] < 1


def test_polyline_helpers():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    pts = resample_polyline(line, np.array([0.0, 0.25, 0.5, 1.0]))
    assert np.allclose(pts, [[0, 0], [0.5, 0], [1, 0], [1, 1]])
    d = distance_to_polyline(np.array([[0.5, 0.2], [2.0, 0.5]]), line)
    assert np.allclose(d, [0.2, 1.0])


def test_contour_requires_wear_surface(small_mesh):
    from tool_wear_demo.core.mesh import Mesh

    tags = np.where(small_mesh.wear_surface_mask, "back", small_mesh.surface_type)
    with pytest.raises(ValueError):
        ToolContour(Mesh(small_mesh.coords, small_mesh.elements, tags))
