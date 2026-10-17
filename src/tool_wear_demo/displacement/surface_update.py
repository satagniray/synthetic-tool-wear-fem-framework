"""Wear-contour node displacement for one cycle."""

from __future__ import annotations

import numpy as np

from ..core.contour import ToolContour, resample_polyline
from ..physics.directions import displacement_vectors, wear_directions


def surface_displacement(
    contour: ToolContour,
    coords: np.ndarray,
    wear_depth_mm: np.ndarray,
    flank_direction: str = "cutting_normal",
    scale: float = 1.0,
) -> np.ndarray:
    """Displacement vectors (mm) of the wear-contour nodes, ordered like the contour."""
    depth = np.asarray(wear_depth_mm, dtype=float)
    if depth.shape != (len(contour),):
        raise ValueError("wear_depth_mm must have one value per contour node")
    if np.any(depth < 0):
        raise ValueError("wear depth cannot be negative")
    dirs = wear_directions(contour, coords, flank_direction)
    return displacement_vectors(depth * float(scale), dirs)


def apply_displacement(coords: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Return ``coords + displacement`` without modifying the input."""
    return np.asarray(coords, dtype=float) + np.asarray(displacement, dtype=float)


def redistribute_surface_nodes(
    contour: ToolContour,
    coords: np.ndarray,
    surface_disp: np.ndarray,
    reference_fractions: np.ndarray,
) -> np.ndarray:
    """Relocate wear-contour nodes tangentially along the worn profile.

    Wear moves the nodes of a convex cutting edge towards each other, which
    crushes the elements there after a few cycles. After the wear step the worn
    profile (moved nodes plus the two fixed anchors) is kept, but the nodes are
    slid along it back to their original normalised arc-length positions. The
    profile shape - and therefore the removed material - is unchanged; only the
    node spacing is restored. Returns the corrected surface displacement.
    """
    moved = contour.points(coords) + surface_disp
    a0, a1 = contour.anchor_index
    polyline = np.vstack([coords[a0], moved, coords[a1]])
    return resample_polyline(polyline, reference_fractions) - contour.points(coords)
