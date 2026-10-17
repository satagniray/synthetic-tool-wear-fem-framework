"""Mesh, geometry and displacement plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ..core.contour import ToolContour
from ..core.mesh import WEAR_SURFACE_TYPES, Mesh
from .style import (
    HEAT_CMAP,
    INK,
    MESH_LINE,
    MUTED,
    SURFACE_COLORS,
    TOOL_FILL,
    new_figure,
    styled,
    tag_synthetic,
    zoom_limits,
)


def draw_mesh(ax, mesh: Mesh, coords: np.ndarray | None = None, color: str = MESH_LINE, lw: float = 0.35, fill: bool = True) -> None:
    xy = mesh.coords if coords is None else coords
    if fill:
        ax.add_collection(PolyCollection(xy[mesh.elements], facecolor=TOOL_FILL, edgecolor="none", zorder=0))
    edges = mesh.unique_edges()
    ax.add_collection(LineCollection(xy[edges], colors=color, linewidths=lw, zorder=1))


def draw_wear_contour(ax, contour: ToolContour, coords: np.ndarray, lw: float = 2.2, by_type: bool = True, **kwargs) -> None:
    """Draw the wear contour, coloured by surface type (or with one style via kwargs)."""
    pts = contour.anchored_points(coords)
    types = np.concatenate([["back"], contour.surface_type, ["back"]])
    if not by_type:
        ax.plot(pts[:, 0], pts[:, 1], lw=lw, **kwargs)
        return
    for kind in WEAR_SURFACE_TYPES:
        mask = types == kind
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        lo, hi = max(idx.min() - 1, 0), min(idx.max() + 1, len(pts) - 1)
        seg = pts[lo : hi + 1]
        ax.plot(seg[:, 0], seg[:, 1], color=SURFACE_COLORS[kind], lw=lw, solid_capstyle="round", zorder=4)


def _surface_legend(ax, loc="upper right", extra=None) -> None:
    handles = [Line2D([], [], color=SURFACE_COLORS[k], lw=3, label=f"{k} face" if k != "edge" else "rounded edge") for k in ("rake", "edge", "flank")]
    handles.append(Line2D([], [], color=SURFACE_COLORS["back"], lw=3, label="clamped back faces"))
    if extra:
        handles.extend(extra)
    ax.legend(handles=handles, loc=loc, fontsize=8)


def plot_initial_mesh(mesh: Mesh, contour: ToolContour, path: str | Path, zoom_mm: float = 0.3) -> Path:
    """Figure 2 - synthetic tool section, full view and cutting-edge close-up."""
    with styled():
        fig = new_figure(12.5, 5.6)
        ax_full, ax_zoom = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
        boundary = contour.boundary_polygon(mesh.coords)
        for ax in (ax_full, ax_zoom):
            draw_mesh(ax, mesh, lw=0.3 if ax is ax_full else 0.5)
            closed = np.vstack([boundary, boundary[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=SURFACE_COLORS["back"], lw=1.6, zorder=3)
            draw_wear_contour(ax, contour, mesh.coords, lw=2.6)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]  (cutting direction)")
            ax.set_ylabel("y [mm]")
        ax_full.autoscale_view()
        ax_full.margins(0.04)
        (x0, x1), (y0, y1) = zoom_limits(zoom_mm)
        ax_full.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=1.0, ls="--", zorder=5))
        ax_full.set_title(f"Synthetic tool section - {mesh.n_elements} quads, {mesh.n_nodes} nodes")

        interior = mesh.surface_type == "interior"
        ax_zoom.scatter(*mesh.coords[interior].T, s=4, color=SURFACE_COLORS["interior"], zorder=2, label="interior nodes")
        for kind in WEAR_SURFACE_TYPES:
            m = mesh.surface_type == kind
            ax_zoom.scatter(*mesh.coords[m].T, s=16, color=SURFACE_COLORS[kind], edgecolor="white", lw=0.4, zorder=5)
        ax_zoom.set_xlim(x0, x1)
        ax_zoom.set_ylim(y0, y1)
        ax_zoom.set_title("Cutting-edge close-up: surface tags and graded mesh")
        _surface_legend(ax_zoom, extra=[Line2D([], [], marker="o", ls="", color=SURFACE_COLORS["interior"], ms=3, label="interior nodes")])
        tag_synthetic(fig)
        fig.savefig(path)
    return Path(path)


def plot_cycle_geometry(
    mesh: Mesh,
    contour: ToolContour,
    path: str | Path,
    zoom_mm: float = 0.3,
    title: str = "",
    previous_coords: np.ndarray | None = None,
    surface_displacement: np.ndarray | None = None,
) -> Path:
    """Per-cycle geometry: mesh near the edge, optional previous contour and magnified motion arrows."""
    with styled():
        fig = new_figure(7.2, 5.0)
        ax = fig.subplots()
        draw_mesh(ax, mesh, lw=0.4)
        if previous_coords is not None:
            prev = contour.anchored_points(previous_coords)
            ax.plot(prev[:, 0], prev[:, 1], color=INK, lw=1.0, ls="--", zorder=5, label="contour before")
        draw_wear_contour(ax, contour, mesh.coords, lw=2.2)
        extra = []
        if surface_displacement is not None and previous_coords is not None:
            mag = np.linalg.norm(surface_displacement, axis=1)
            if mag.max() > 0:
                (x0, x1), _ = zoom_limits(zoom_mm)
                factor = 0.06 * (x1 - x0) / mag.max()
                base = contour.points(previous_coords)
                show = mag > 0.05 * mag.max()
                ax.quiver(
                    base[show, 0], base[show, 1], surface_displacement[show, 0] * factor, surface_displacement[show, 1] * factor,
                    angles="xy", scale_units="xy", scale=1.0, color=INK, width=0.003, zorder=6,
                )
                extra.append(Line2D([], [], color=INK, marker=r"$\rightarrow$", ls="", ms=12, label=f"node motion (x{factor:,.0f}, max {mag.max() * 1000:.2f} um)"))
                extra.append(Line2D([], [], color=INK, ls="--", lw=1.0, label="contour before"))
        (x0, x1), (y0, y1) = zoom_limits(zoom_mm)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(title)
        _surface_legend(ax, extra=extra)
        tag_synthetic(fig)
        fig.savefig(path)
    return Path(path)


def plot_displacement_heatmap(
    mesh: Mesh,
    contour: ToolContour,
    displacement_um: np.ndarray,
    path: str | Path,
    zoom_mm: float = 0.3,
    reference_coords: np.ndarray | None = None,
    title: str = "Cumulative node displacement",
) -> Path:
    """Figure 6 - displacement magnitude of surface and interior nodes (element average)."""
    mag = np.linalg.norm(displacement_um, axis=1) if displacement_um.ndim == 2 else displacement_um
    elem_val = mag[mesh.elements].mean(axis=1)
    with styled():
        fig = new_figure(8.4, 5.4)
        ax = fig.subplots()
        pc = PolyCollection(mesh.coords[mesh.elements], array=elem_val, cmap=HEAT_CMAP, edgecolor=(1, 1, 1, 0.35), linewidth=0.25)
        pc.set_clim(0.0, max(float(mag.max()), 1e-9))
        ax.add_collection(pc)
        if reference_coords is not None:
            ref = contour.anchored_points(reference_coords)
            ax.plot(ref[:, 0], ref[:, 1], color=INK, lw=1.0, ls="--", label="unworn contour")
            ax.legend(loc="upper right", fontsize=8)
        cur = contour.anchored_points(mesh.coords)
        ax.plot(cur[:, 0], cur[:, 1], color=INK, lw=1.4)
        cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("|u| [um]")
        (x0, x1), (y0, y1) = zoom_limits(zoom_mm)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(title)
        ax.grid(False)
        tag_synthetic(fig)
        fig.savefig(path)
    return Path(path)
