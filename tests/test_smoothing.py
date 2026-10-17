import numpy as np
import pytest

from tool_wear_demo.displacement import (
    LaplacianSmoother,
    NormalizedRadialSmoother,
    redistribute_surface_nodes,
    surface_displacement,
)
from tool_wear_demo.physics.directions import CUTTING_NORMAL, wear_directions


def _surface_disp(contour, mesh, amplitude_mm=0.002):
    depth = amplitude_mm * np.exp(-((contour.signed_edge_distance(mesh.coords) / 0.08) ** 2))
    return surface_displacement(contour, mesh.coords, depth)


@pytest.mark.parametrize("smoother", [NormalizedRadialSmoother(0.08), LaplacianSmoother(300)])
def test_interior_never_exceeds_surface_max(small_mesh, contour, smoother):
    surf = _surface_disp(contour, small_mesh)
    full = smoother.smooth(small_mesh, contour.node_index, surf)
    interior = small_mesh.surface_type == "interior"
    assert np.linalg.norm(full[interior], axis=1).max() <= np.linalg.norm(surf, axis=1).max() * (1 + 1e-12)


@pytest.mark.parametrize("smoother", [NormalizedRadialSmoother(0.08), LaplacianSmoother(300)])
def test_fixed_nodes_stay_fixed_and_surface_is_prescribed(small_mesh, contour, smoother):
    surf = _surface_disp(contour, small_mesh)
    full = smoother.smooth(small_mesh, contour.node_index, surf)
    assert np.all(full[small_mesh.fixed_mask] == 0.0)
    assert np.allclose(full[contour.node_index], surf)


def test_overlapping_neighbourhoods_do_not_amplify(small_mesh, contour):
    """Uniform surface motion must not add up in the interior."""
    surf = np.tile([0.0, 0.003], (len(contour), 1))
    full = NormalizedRadialSmoother(0.5).smooth(small_mesh, contour.node_index, surf)
    assert np.linalg.norm(full, axis=1).max() <= 0.003 * (1 + 1e-12)


def test_radial_displacement_decays_with_distance(small_mesh, contour):
    surf = _surface_disp(contour, small_mesh)
    smoother = NormalizedRadialSmoother(0.08)
    full = smoother.smooth(small_mesh, contour.node_index, surf)
    src = small_mesh.coords[contour.node_index]
    dist = np.linalg.norm(small_mesh.coords[:, None] - src[None], axis=2).min(axis=1)
    interior = small_mesh.surface_type == "interior"
    assert np.all(full[interior & (dist >= 0.08)] == 0.0)
    mag = np.linalg.norm(full, axis=1)
    near = interior & (dist < 0.03) & (mag > 0)
    far = interior & (dist >= 0.05) & (dist < 0.08)
    assert near.any() and far.any()
    assert mag[near].mean() > mag[far].mean()


def test_wear_directions_point_into_tool_with_requested_magnitude(small_mesh, contour):
    for mode in ("cutting_normal", "surface_normal"):
        dirs = wear_directions(contour, small_mesh.coords, mode)
        normals = contour.inward_normals(small_mesh.coords)
        assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
        assert np.all(np.sum(dirs * normals, axis=1) > 0)
    depth = np.linspace(0.0, 0.003, len(contour))
    disp = surface_displacement(contour, small_mesh.coords, depth)
    assert np.allclose(np.linalg.norm(disp, axis=1), depth)


def test_flank_moves_perpendicular_to_cutting_velocity(small_mesh, contour):
    dirs = wear_directions(contour, small_mesh.coords, "cutting_normal")
    assert np.allclose(dirs[contour.mask("flank")], CUTTING_NORMAL)
    edge = dirs[contour.mask("edge")]
    assert np.allclose(edge[0], CUTTING_NORMAL)  # blend starts at the flank direction


def test_zero_depth_gives_zero_motion(small_mesh, contour):
    disp = surface_displacement(contour, small_mesh.coords, np.zeros(len(contour)))
    assert np.all(disp == 0)
    with pytest.raises(ValueError):
        surface_displacement(contour, small_mesh.coords, -np.ones(len(contour)))


def test_redistribution_restores_spacing_without_changing_shape(small_mesh, contour):
    fr = contour.anchored_fractions(small_mesh.coords)
    surf = _surface_disp(contour, small_mesh, amplitude_mm=0.004)
    moved = contour.points(small_mesh.coords) + surf
    corrected = redistribute_surface_nodes(contour, small_mesh.coords, surf, fr)
    new_pts = contour.points(small_mesh.coords) + corrected
    new_coords = small_mesh.coords.copy()
    new_coords[contour.node_index] = new_pts
    # spacing restored (up to the tiny chord-vs-arc difference of the resampled polyline)
    assert np.allclose(contour.anchored_fractions(new_coords), fr, atol=1e-4)
    from tool_wear_demo.core.contour import distance_to_polyline

    a0, a1 = contour.anchor_index
    worn_profile = np.vstack([small_mesh.coords[a0], moved, small_mesh.coords[a1]])
    assert distance_to_polyline(new_pts, worn_profile).max() < 1e-9
