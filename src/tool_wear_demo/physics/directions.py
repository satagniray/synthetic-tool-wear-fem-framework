"""Direction in which each wear-contour node recedes."""

from __future__ import annotations

import numpy as np

from ..core.contour import ToolContour

#: Unit normal of the cutting velocity pointing into the tool (cutting velocity is along x).
CUTTING_NORMAL = np.array([0.0, 1.0])


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def wear_directions(contour: ToolContour, coords: np.ndarray, flank_direction: str = "cutting_normal") -> np.ndarray:
    """Unit recession directions for every contour node, pointing into the tool.

    * rake nodes: local inward surface normal;
    * flank nodes: ``cutting_normal`` -> perpendicular to the cutting velocity
      (produces the flat flank wear land seen in practice), or
      ``surface_normal`` -> local inward normal;
    * edge nodes: blend from the flank direction (flank side) to the local
      inward normal (rake side) using the position along the rounded edge.
    """
    normals = contour.inward_normals(coords)
    dirs = normals.copy()
    if flank_direction == "cutting_normal":
        flank = contour.mask("flank")
        dirs[flank] = CUTTING_NORMAL
        edge = contour.mask("edge")
        phi = contour.edge_fraction(coords)[edge][:, None]
        dirs[edge] = _normalize((1.0 - phi) * CUTTING_NORMAL + phi * normals[edge])
    elif flank_direction != "surface_normal":
        raise ValueError(f"unknown flank_direction '{flank_direction}'")
    outward = np.sum(dirs * normals, axis=1) <= 1e-9  # safety: never move out of the tool
    dirs[outward] = normals[outward]
    return dirs


def displacement_vectors(wear_depth_mm: np.ndarray, unit_directions: np.ndarray) -> np.ndarray:
    """``displacement = wear_depth * unit_direction`` (mm)."""
    return np.asarray(wear_depth_mm, dtype=float)[:, None] * unit_directions
