"""Element quality metrics for 2D quadrilaterals and the accept/reject check."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def signed_areas(coords: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Shoelace signed area of every quad (positive for counter-clockwise nodes)."""
    p = coords[elements]  # (M, 4, 2)
    x, y = p[..., 0], p[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


def corner_jacobians(coords: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corner Jacobians and scaled Jacobians (sine of the corner angle), shape ``(M, 4)``.

    A non-positive corner Jacobian means the element is inverted or non-convex.
    """
    p = coords[elements]
    e_next = np.roll(p, -1, axis=1) - p
    e_prev = np.roll(p, 1, axis=1) - p
    jac = e_next[..., 0] * e_prev[..., 1] - e_next[..., 1] * e_prev[..., 0]
    norms = np.linalg.norm(e_next, axis=2) * np.linalg.norm(e_prev, axis=2)
    scaled = np.divide(jac, norms, out=np.zeros_like(jac), where=norms > 0)
    return jac, scaled


def element_metrics(coords: np.ndarray, elements: np.ndarray) -> dict[str, np.ndarray]:
    """Per-element area, scaled Jacobian, minimum angle, edge lengths and aspect ratio."""
    p = coords[elements]
    e_next = np.roll(p, -1, axis=1) - p
    e_prev = np.roll(p, 1, axis=1) - p
    jac, scaled = corner_jacobians(coords, elements)
    dots = np.sum(e_next * e_prev, axis=2)
    angles = np.degrees(np.arctan2(jac, dots))  # negative -> folded corner
    lengths = np.linalg.norm(e_next, axis=2)
    min_len = lengths.min(axis=1)
    aspect = np.divide(lengths.max(axis=1), min_len, out=np.full(min_len.shape, np.inf), where=min_len > 0)
    return {
        "area": signed_areas(coords, elements),
        "min_scaled_jacobian": scaled.min(axis=1),
        "min_angle_deg": angles.min(axis=1),
        "min_edge_length": min_len,
        "aspect_ratio": aspect,
    }


@dataclass
class QualityReport:
    """Structured result of a mesh-quality check (written to ``mesh_quality.json``)."""

    valid: bool
    n_elements: int
    n_invalid: int
    min_area: float
    min_quality: float  # minimum scaled Jacobian
    min_angle_deg: float
    max_aspect_ratio: float
    min_edge_length: float
    retries: int = 0
    displacement_scale: float = 1.0

    def to_dict(self) -> dict:
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in asdict(self).items()}


class MeshQualityChecker:
    """Reject meshes with inverted/folded elements or overly distorted corners.

    Parameters
    ----------
    min_scaled_jacobian:
        Smallest acceptable corner scaled Jacobian (sine of the corner angle).
        ``0`` only rejects inverted elements.
    """

    def __init__(self, min_scaled_jacobian: float = 0.0) -> None:
        self.min_scaled_jacobian = float(min_scaled_jacobian)

    def check(self, coords: np.ndarray, elements: np.ndarray) -> QualityReport:
        m = element_metrics(coords, elements)
        bad = (m["area"] <= 0.0) | (m["min_scaled_jacobian"] <= 0.0)
        bad |= m["min_scaled_jacobian"] < self.min_scaled_jacobian
        return QualityReport(
            valid=bool(not bad.any()),
            n_elements=int(elements.shape[0]),
            n_invalid=int(bad.sum()),
            min_area=float(m["area"].min()),
            min_quality=float(m["min_scaled_jacobian"].min()),
            min_angle_deg=float(m["min_angle_deg"].min()),
            max_aspect_ratio=float(m["aspect_ratio"].max()),
            min_edge_length=float(m["min_edge_length"].min()),
        )
