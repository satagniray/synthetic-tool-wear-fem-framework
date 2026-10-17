"""Shared figure style (object-oriented Matplotlib - no pyplot, no GUI backend needed)."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE_COLORS = {
    "flank": "#C8553D",
    "edge": "#E8A33D",
    "rake": "#2E86AB",
    "back": "#4A4E69",
    "interior": "#B9BEC7",
}
INK = "#1F2430"
MUTED = "#6B7280"
GRID = "#E5E7EB"
MESH_LINE = "#9AA3AF"
TOOL_FILL = "#DCE3EA"
ACCENT = "#C8553D"

#: light steel -> deep navy, used for "cycle number" colouring
CYCLE_CMAP = LinearSegmentedColormap.from_list("cycles", ["#9CC3D5", "#2E86AB", "#1B3B6F", "#0B1D3A"])
#: displacement heat map
HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#F7F7F2", "#F2C14E", "#E07A3F", "#B23A48", "#4A1942"])

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#9AA3AF",
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

SYNTHETIC_TAG = "synthetic data - portfolio demonstrator, not experimentally validated"


@contextmanager
def styled():
    with mpl.rc_context(RC):
        yield


def new_figure(width: float, height: float, **kwargs) -> Figure:
    return Figure(figsize=(width, height), layout="constrained", **kwargs)


def tag_synthetic(fig: Figure) -> None:
    fig.text(0.995, -0.005, SYNTHETIC_TAG, ha="right", va="top", fontsize=7, color=MUTED, style="italic")


def zoom_limits(zoom_mm: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Window around the cutting edge (tip at the origin)."""
    return (-0.04, zoom_mm), (-0.03, 0.72 * zoom_mm)
