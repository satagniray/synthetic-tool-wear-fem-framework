"""Public-demo wear metrics.

All metrics are *geometric indicators* computed on the synthetic mesh. In
particular the flank-wear indicator is NOT an ISO 3685 VB measurement and is
not experimentally validated.
"""

from __future__ import annotations

import numpy as np

from ..core.contour import ToolContour, distance_to_polyline
from ..core.mesh import Mesh


def surface_recession_um(reference: Mesh, current: Mesh, contour: ToolContour) -> np.ndarray:
    """Distance from each *original* contour point to the *current* worn profile (um)."""
    return 1000.0 * distance_to_polyline(contour.points(reference.coords), contour.anchored_points(current.coords))


def flank_wear_indicator_mm(reference: Mesh, recession_um: np.ndarray, contour: ToolContour, threshold_um: float) -> float:
    """Approximate geometric flank-wear indicator ("VB-like"), mm.

    Horizontal extent (along the cutting direction) of the worn region on the
    flank side of the edge - original contour points with ``d <= 0`` whose
    recession exceeds ``threshold_um`` - measured from the lowest point of the
    unworn rounded edge.
    """
    pts = contour.points(reference.coords)
    d = contour.signed_edge_distance(reference.coords)
    worn = (d <= 0.0) & (recession_um > threshold_um)
    if not worn.any():
        return 0.0
    x_origin = pts[np.argmin(pts[:, 1]), 0]
    return float(max(0.0, pts[worn, 0].max() - x_origin))


def compute_wear_metrics(reference: Mesh, current: Mesh, contour: ToolContour, vb_threshold_um: float) -> dict[str, float]:
    rec = surface_recession_um(reference, current, contour)
    types = contour.surface_type
    worn = rec > vb_threshold_um

    def _mean(mask: np.ndarray) -> float:
        return float(rec[mask].mean()) if mask.any() else 0.0

    return {
        "vb_indicator_mm": flank_wear_indicator_mm(reference, rec, contour, vb_threshold_um),
        "max_recession_um": float(rec.max()),
        "crater_depth_indicator_um": float(rec[types == "rake"].max()) if (types == "rake").any() else 0.0,
        "edge_recession_um": float(rec[types == "edge"].max()) if (types == "edge").any() else 0.0,
        "mean_rake_recession_um": _mean((types == "rake") & worn),
        "mean_flank_recession_um": _mean((types == "flank") & worn),
        "removed_area_um2": float((reference.area() - current.area()) * 1.0e6),
    }
