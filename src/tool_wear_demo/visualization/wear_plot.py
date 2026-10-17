"""Process-field, wear-progression, wear-indicator and workflow figures."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..core.contour import ToolContour
from ..core.mesh import Mesh
from ..core.state import ProcessFieldState
from .mesh_plot import draw_mesh
from .style import (
    ACCENT,
    CYCLE_CMAP,
    INK,
    MUTED,
    SURFACE_COLORS,
    TOOL_FILL,
    new_figure,
    styled,
    tag_synthetic,
    zoom_limits,
)


def _shade_regions(ax, d_mm: np.ndarray, types: np.ndarray, window_um: tuple[float, float], label: bool = False) -> None:
    edge = types == "edge"
    if not edge.any():
        return
    lo, hi = d_mm[edge].min() * 1000, d_mm[edge].max() * 1000
    x_min, x_max = window_um
    for a, b, kind in ((x_min, lo, "flank face"), (lo, hi, "edge"), (hi, x_max, "rake face")):
        ax.axvspan(a, b, color=SURFACE_COLORS[kind.split()[0]], alpha=0.08, lw=0)
        if label:
            ax.text((a + b) / 2, 1.02, kind, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=SURFACE_COLORS[kind.split()[0]], fontweight="bold")


def plot_process_fields(
    fields: ProcessFieldState,
    path: str | Path,
    wear_rate: np.ndarray | None = None,
    title: str = "Synthetic process fields along the tool surface",
    window_mm: tuple[float, float] = (-0.30, 0.40),
) -> Path:
    """Figure 3 - pressure, temperature, sliding velocity (and wear rate) vs. surface position."""
    d_um = fields.d_mm * 1000.0
    rows = [
        (fields.pressure_MPa, "contact pressure\n$\\sigma_n$ [MPa]", "#2E86AB"),
        (fields.temperature_K, "interface temp.\n$\\theta$ [K]", "#C8553D"),
        (fields.sliding_velocity_m_s, "sliding velocity\n$v_s$ [m/s]", "#4A4E69"),
    ]
    if wear_rate is not None:
        rows.append((np.asarray(wear_rate) * 1000.0 * 60.0, "wear rate\n$\\dot w$ [um/min]", "#7A3B69"))
    with styled():
        fig = new_figure(8.6, 1.75 * len(rows) + 0.9)
        axes = np.atleast_1d(fig.subplots(len(rows), 1, sharex=True))
        window_um = (window_mm[0] * 1000, window_mm[1] * 1000)
        for i, (ax, (values, label, color)) in enumerate(zip(axes, rows)):
            _shade_regions(ax, fields.d_mm, fields.surface_type, window_um, label=(i == 0))
            ax.plot(d_um, values, color=color, lw=1.8)
            ax.plot(d_um, values, "o", color=color, ms=2.4)
            ax.set_ylabel(label)
        axes[-1].set_xlabel("signed distance from the cutting-edge middle  d [um]   (flank < 0 < rake)")
        axes[-1].set_xlim(window_mm[0] * 1000, window_mm[1] * 1000)
        fig.suptitle(title, fontweight="bold")
        tag_synthetic(fig)
        fig.savefig(path)
    return Path(path)


def _cycle_colors(n: int) -> list:
    return [CYCLE_CMAP(x) for x in np.linspace(0.05, 1.0, max(n, 1))]


def plot_wear_progression(
    mesh: Mesh,
    contour: ToolContour,
    snapshots: Sequence[tuple[int, np.ndarray]],
    path: str | Path,
    zoom_mm: float = 0.3,
    highlight: Sequence[int] | None = None,
) -> Path:
    """Figure 4 - overlay of the tool contour at selected cycles."""
    snaps = dict(snapshots)
    cycles = sorted(snaps)
    if highlight is None:
        final = cycles[-1]
        highlight = sorted({c for c in (0, 5, 10, final) if c in snaps})
    with styled():
        fig = new_figure(8.6, 5.6)
        ax = fig.subplots()
        final_coords = snaps[cycles[-1]]
        draw_mesh(ax, mesh, coords=final_coords, lw=0.25, color="#C3CAD3")
        for c in cycles:  # faint trail of every cycle
            pts = contour.anchored_points(snaps[c])
            ax.plot(pts[:, 0], pts[:, 1], color=CYCLE_CMAP(0.15 + 0.85 * c / max(cycles[-1], 1)), lw=0.6, alpha=0.45)
        handles = []
        for c, color in zip(highlight, _cycle_colors(len(highlight))):
            pts = contour.anchored_points(snaps[c])
            style = dict(ls="--", lw=1.6) if c == 0 else dict(ls="-", lw=2.2)
            ax.plot(pts[:, 0], pts[:, 1], color=INK if c == 0 else color, zorder=5, **style)
            handles.append(Line2D([], [], color=INK if c == 0 else color, label=f"cycle {c}" + (" (unworn)" if c == 0 else ""), **style))
        (x0, x1), (y0, y1) = zoom_limits(zoom_mm)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Tool-contour progression: crater on the rake face, wear land on the flank")
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        tag_synthetic(fig)
        fig.savefig(path)
    return Path(path)


def plot_wear_indicator(history: Sequence[Mapping], path: str | Path, target_mm: float | None = None) -> Path:
    """Figure 5 - flank-wear indicator and recession metrics vs. cycle."""
    cyc = np.array([h["cycle"] for h in history], dtype=float)
    t_min = np.array([h["elapsed_time_s"] for h in history], dtype=float) / 60.0
    vb = np.array([h["vb_indicator_mm"] for h in history], dtype=float) * 1000.0
    crater = np.array([h["crater_depth_indicator_um"] for h in history], dtype=float)
    edge = np.array([h["edge_recession_um"] for h in history], dtype=float)
    flank = np.array([h["mean_flank_recession_um"] for h in history], dtype=float)
    with styled():
        fig = new_figure(11.5, 4.3)
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(cyc, vb, "-o", color=SURFACE_COLORS["flank"], lw=2, ms=4, label="flank-wear indicator")
        if target_mm is not None:
            ax1.axhline(target_mm * 1000, color=INK, ls="--", lw=1)
            ax1.text(cyc[0], target_mm * 1000, "  stopping criterion", va="bottom", fontsize=8, color=INK)
        ax1.set_xlabel("wear cycle")
        ax1.set_ylabel("approx. geometric flank-wear indicator [um]")
        ax1.set_title("Flank-wear indicator vs. cycle")
        top = ax1.twiny()
        top.set_xlim(ax1.get_xlim())
        step = max(1, int(np.ceil(len(cyc) / 8)))
        top.set_xticks(cyc[::step])
        top.set_xticklabels([f"{t:.0f}" for t in t_min[::step]])
        top.grid(False)
        top.set_xlabel("equivalent cutting time [min] (synthetic)", color=MUTED, fontsize=8)
        ax2.plot(cyc, crater, "-o", color=SURFACE_COLORS["rake"], lw=2, ms=4, label="crater depth (rake, max)")
        ax2.plot(cyc, edge, "-s", color=SURFACE_COLORS["edge"], lw=2, ms=4, label="edge recession (max)")
        ax2.plot(cyc, flank, "-^", color=SURFACE_COLORS["flank"], lw=2, ms=4, label="flank recession (mean of worn)")
        ax2.set_xlabel("wear cycle")
        ax2.set_ylabel("surface recession [um]")
        ax2.set_title("Recession indicators vs. cycle")
        ax2.legend(fontsize=8, loc="upper left")
        tag_synthetic(fig)
        fig.savefig(path)
    return Path(path)


def plot_workflow_diagram(path: str | Path) -> Path:
    """Figure 1 - the automated wear loop."""
    steps = [
        ("Synthetic process fields", "contact pressure, temperature, sliding velocity\n(solver backend: synthetic | Abaqus stub)"),
        ("Wear model", "Usui-type law  w = C . sigma . v_s . exp(-lambda / theta)"),
        ("Surface displacement", "wear depth x unit direction  +  node relocation"),
        ("Interior smoothing", "normalised radial weights, bounded motion"),
        ("Mesh quality check", "signed area, scaled Jacobian, angles"),
        ("Updated tool geometry", "metrics, CSV/JSON/PNG output, checkpoint"),
    ]
    colors = ["#2E86AB", "#7A3B69", "#C8553D", "#E8A33D", "#4A4E69", "#1B3B6F"]
    with styled():
        fig = new_figure(8.2, 8.0)
        ax = fig.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10.6)
        ax.axis("off")
        w, h = 6.0, 1.05
        x = 2.0
        ys = [9.1 - i * 1.55 for i in range(len(steps))]
        for (head, sub), y, c in zip(steps, ys, colors):
            ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc="white", ec=c, lw=2.0))
            ax.add_patch(FancyBboxPatch((x, y - h / 2), 0.16, h, boxstyle="square,pad=0", fc=c, ec=c))
            ax.text(x + 0.45, y + 0.16, head, fontsize=12, fontweight="bold", color=INK, va="center")
            ax.text(x + 0.45, y - 0.24, sub, fontsize=8.2, color=MUTED, va="center")
        for y0, y1 in zip(ys[:-1], ys[1:]):
            ax.add_patch(FancyArrowPatch((x + w / 2, y0 - h / 2), (x + w / 2, y1 + h / 2), arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.4))
        # next-cycle loop
        ax.add_patch(
            FancyArrowPatch((x, ys[-1]), (x, ys[0]), connectionstyle="arc3,rad=-0.45", arrowstyle="-|>", mutation_scale=18, color=ACCENT, lw=2.0)
        )
        ax.text(0.75, (ys[0] + ys[-1]) / 2, "next cycle\nuntil VB target\nor max cycles", color=ACCENT, fontsize=9, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))
        # retry loop
        ax.add_patch(
            FancyArrowPatch((x + w, ys[4]), (x + w, ys[2]), connectionstyle="arc3,rad=0.6", arrowstyle="-|>", mutation_scale=14, color="#4A4E69", lw=1.4, ls="--")
        )
        ax.text(x + w + 1.1, ys[3], "invalid?\nscale x 0.5\nand retry", color="#4A4E69", fontsize=8.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))
        ax.text(5.0, 10.35, "Automated multi-cycle tool-wear loop", ha="center", fontsize=14, fontweight="bold", color=INK)
        fig.savefig(path)
    return Path(path)


def render_progression_frames(
    mesh: Mesh,
    contour: ToolContour,
    snapshots: Sequence[tuple[int, np.ndarray]],
    history: Sequence[Mapping],
    zoom_mm: float = 0.3,
):
    """Yield PIL images, one per cycle, for the progression GIF."""
    from PIL import Image

    by_cycle = {int(h["cycle"]): h for h in history}
    ref = dict(snapshots)[0]
    last = max(c for c, _ in snapshots)
    with styled():
        for c, coords in snapshots:
            fig = new_figure(7.2, 4.8, dpi=100)
            ax = fig.subplots()
            draw_mesh(ax, mesh, coords=coords, lw=0.3, color="#B7C0CC")
            r = contour.anchored_points(ref)
            ax.plot(r[:, 0], r[:, 1], color=INK, ls="--", lw=1.0)
            p = contour.anchored_points(coords)
            ax.plot(p[:, 0], p[:, 1], color=CYCLE_CMAP(0.2 + 0.8 * c / max(last, 1)), lw=2.6)
            (x0, x1), (y0, y1) = zoom_limits(zoom_mm)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            h = by_cycle.get(c, {})
            vb = float(h.get("vb_indicator_mm", 0.0)) * 1000
            t = float(h.get("elapsed_time_s", 0.0)) / 60
            ax.set_title(f"cycle {c:2d}   |   t = {t:5.1f} min   |   flank-wear indicator = {vb:5.1f} um")
            ax.text(0.99, 0.01, "synthetic data - not experimentally validated", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED, style="italic")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches=None)
            buf.seek(0)
            yield Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def make_wear_gif(frames, path: str | Path, frame_ms: int = 380, hold_last_ms: int = 1600) -> Path:
    frames = list(frames)
    durations = [frame_ms] * len(frames)
    durations[-1] = hold_last_ms
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return Path(path)
