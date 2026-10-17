"""Build docs/project_overview.pdf - a 7-slide technical overview (16:9).

Uses only Matplotlib. Expects the figures in docs/images (created by
scripts/build_docs_assets.py).

    python scripts/build_project_overview.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2430"
MUTED = "#6B7280"
NAVY = "#1B3B6F"
RAKE = "#2E86AB"
EDGE = "#E8A33D"
FLANK = "#C8553D"
PLUM = "#7A3B69"
SLATE = "#4A4E69"
W, H = 13.333, 7.5
TOTAL = 7


def _slide(pdf: PdfPages, number: int, title: str, kicker: str) -> tuple[Figure, object]:
    fig = Figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, H - 0.09), W, 0.09, color=NAVY))
    ax.text(0.6, H - 0.55, kicker.upper(), fontsize=10, color=FLANK, fontweight="bold")
    ax.text(0.6, H - 1.05, title, fontsize=24, color=INK, fontweight="bold")
    ax.plot([0.6, W - 0.6], [H - 1.35, H - 1.35], color="#E5E7EB", lw=1.2)
    ax.text(0.6, 0.3, "Synthetic Multi-Cycle Tool-Wear Simulation Framework  |  independent portfolio demonstrator  |  synthetic data, not experimentally validated",
            fontsize=8, color=MUTED)
    ax.text(W - 0.6, 0.3, f"{number} / {TOTAL}", fontsize=9, color=MUTED, ha="right")
    return fig, ax


def _image(fig: Figure, path: Path, rect: tuple[float, float, float, float]) -> None:
    """Place an image inside rect = (x, y, w, h) in inches, preserving aspect ratio."""
    img = imread(path)
    x, y, w, h = rect
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    ax = fig.add_axes([(x + (w - dw) / 2) / W, (y + (h - dh) / 2) / H, dw / W, dh / H])
    ax.imshow(img)
    ax.axis("off")


def _bullets(ax, x: float, y: float, items: list[str], size: float = 13, gap: float = 0.55, width_chars: int = 46) -> float:
    import textwrap

    for item in items:
        lines = textwrap.wrap(item, width_chars)
        ax.add_patch(Rectangle((x, y - 0.03), 0.09, 0.09, color=FLANK))
        for k, line in enumerate(lines):
            ax.text(x + 0.25, y - k * 0.32, line, fontsize=size, color=INK, va="center")
        y -= gap + 0.32 * (len(lines) - 1)
    return y


def _cutting_schematic(ax, x0: float, y0: float, s: float) -> None:
    """Simple orthogonal-cutting sketch (workpiece, chip, tool) in slide coordinates."""
    def P(pts):
        return [(x0 + px * s, y0 + py * s) for px, py in pts]

    # workpiece with the uncut layer (left of the tool) and the machined surface (right)
    ax.add_patch(Polygon(P([(0, 0), (6.2, 0), (6.2, 1.6), (3.0, 1.6), (2.3, 2.1), (0, 2.1)]), closed=True, fc="#E7ECF2", ec=SLATE, lw=1.2))
    ax.plot(*zip(*P([(3.0, 1.6), (2.3, 2.1)])), color=SLATE, lw=1, ls=":")  # shear plane
    # chip flowing up the rake face
    ax.add_patch(Polygon(P([(3.0, 1.6), (3.33, 4.1), (2.7, 4.1), (2.45, 3.0), (2.3, 2.1)]), closed=True, fc="#F2C14E", ec="#B07D1A", lw=1.2))
    # tool
    ax.add_patch(Polygon(P([(3.0, 1.6), (3.35, 4.3), (5.3, 4.3), (5.3, 1.85)]), closed=True, fc="#C9D3DD", ec=INK, lw=1.4))
    ax.plot(*zip(*P([(3.0, 1.6), (3.35, 4.3)])), color=RAKE, lw=4, solid_capstyle="round")
    ax.plot(*zip(*P([(3.0, 1.6), (5.3, 1.85)])), color=FLANK, lw=4, solid_capstyle="round")
    ax.add_patch(Wedge((x0 + 3.02 * s, y0 + 1.63 * s), 0.12 * s, 0, 360, color=EDGE))
    ax.add_patch(FancyArrowPatch((x0 + 0.4 * s, y0 + 0.75 * s), (x0 + 2.0 * s, y0 + 0.75 * s), arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.6))
    ax.text(x0 + 1.2 * s, y0 + 0.95 * s, "workpiece moves", fontsize=10, ha="center", color=INK)
    ax.text(x0 + 2.75 * s, y0 + 4.3 * s, "chip", fontsize=10, color="#8A5A00", ha="center", fontweight="bold")
    ax.text(x0 + 4.3 * s, y0 + 3.3 * s, "tool", fontsize=12, color=INK, ha="center", fontweight="bold")
    ax.text(x0 + 3.5 * s, y0 + 2.8 * s, "rake face\n(crater wear)", fontsize=9, color=RAKE, fontweight="bold")
    ax.text(x0 + 4.4 * s, y0 + 1.35 * s, "flank face (flank wear land)", fontsize=9, color=FLANK, fontweight="bold", ha="center")
    ax.text(x0 + 4.5 * s, y0 + 0.3 * s, "machined surface", fontsize=9, color=MUTED, ha="center")


def _history_stats(images: Path) -> dict:
    path = images / "history_default_run.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    last = rows[-1]
    return {
        "cycles": int(last["cycle"]),
        "vb": float(last["vb_indicator_mm"]) * 1000,
        "crater": float(last["crater_depth_indicator_um"]),
        "time": float(last["elapsed_time_s"]) / 60,
        "q": min(float(r["min_scaled_jacobian"]) for r in rows),
    }


def build(out: Path, images: Path) -> Path:
    stats = _history_stats(images)
    with mpl.rc_context({"font.family": "DejaVu Sans"}), PdfPages(out) as pdf:
        # 1 - problem -------------------------------------------------------
        fig, ax = _slide(pdf, 1, "Automating incremental tool-wear prediction", "the problem")
        _bullets(ax, 0.7, 5.4, [
            "Tool wear changes the cutting-edge geometry - and with it forces, temperatures and part quality.",
            "Wear experiments are slow and expensive; FEM can predict wear, but only if the geometry is updated cycle by cycle.",
            "Doing that by hand (simulate, extract fields, move nodes, fix the mesh, re-run) does not scale.",
            "This project automates the full loop in Python - solver-free, so anyone can run and inspect it.",
        ], width_chars=50)
        _cutting_schematic(ax, 7.2, 1.2, 0.95)
        pdf.savefig(fig)

        # 2 - workflow ------------------------------------------------------
        fig, ax = _slide(pdf, 2, "One automated loop, repeated until a wear criterion", "workflow")
        _image(fig, images / "fig1_workflow.png", (0.6, 0.6, 6.4, 5.6))
        _bullets(ax, 7.4, 5.4, [
            "Process fields on the tool surface come from a pluggable backend (synthetic by default).",
            "A literature wear law turns local pressure, temperature and sliding speed into a wear rate.",
            "Surface nodes recede; interior nodes follow; the mesh is checked and the step halved if needed.",
            "Each cycle writes CSV/JSON/PNG and a checkpoint - interrupted runs resume bit-for-bit.",
        ], width_chars=44)
        pdf.savefig(fig)

        # 3 - architecture ---------------------------------------------------
        fig, ax = _slide(pdf, 3, "Software architecture: physics, geometry and solver kept apart", "design")
        boxes = [
            ("io", "YAML config, CSV, run folders", 0.7, 4.4, SLATE),
            ("backends", "SolverBackend protocol\nSyntheticBackend | Abaqus stub", 3.75, 4.4, RAKE),
            ("physics", "synthetic fields, Usui wear law,\nwear directions", 6.8, 4.4, PLUM),
            ("core", "Mesh, ToolContour,\nMeshQualityChecker", 9.85, 4.4, NAVY),
            ("displacement", "surface update, node relocation,\ninterior smoothing", 2.2, 2.2, FLANK),
            ("workflow", "CycleRunner, WearLoop,\ncheckpoints, metrics", 5.3, 2.2, "#1B3B6F"),
            ("visualization", "figures, GIF,\nHTML dashboard", 8.4, 2.2, EDGE),
        ]
        for name, desc, x, y, c in boxes:
            ax.add_patch(FancyBboxPatch((x, y), 2.8, 1.55, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=c, lw=2))
            ax.add_patch(Rectangle((x, y + 1.55 - 0.12), 2.8, 0.12, color=c))
            ax.text(x + 0.2, y + 1.1, name, fontsize=15, fontweight="bold", color=INK, family="DejaVu Sans Mono")
            ax.text(x + 0.2, y + 0.55, desc, fontsize=10, color=MUTED, va="center")
        for x in (2.1, 5.15, 8.2, 11.25):
            ax.add_patch(FancyArrowPatch((x, 4.4), (6.7, 3.78), arrowstyle="-|>", mutation_scale=12, color="#9AA3AF", lw=1))
        ax.text(0.7, 1.35, "src/tool_wear_demo/  -  standard Python (NumPy, Matplotlib, PyYAML); no commercial solver needed", fontsize=12, color=INK)
        ax.text(0.7, 0.85, "74 unit + integration tests, GitHub Actions on Python 3.11 / 3.12, deterministic seeded runs, CLI + examples",
                fontsize=12, color=INK)
        pdf.savefig(fig)

        # 4 - wear model ------------------------------------------------------
        fig, ax = _slide(pdf, 4, "Wear law and synthetic process fields", "physics")
        ax.text(0.7, 5.5, r"$\dot{w} \;=\; C\;\sigma_n\;v_s\;\exp\!\left(-\dfrac{\lambda}{\theta}\right)$", fontsize=26, color=INK)
        ax.text(0.7, 4.75, "Usui-type wear law as used for FEM wear estimation\n(Xie et al., Wear 258, 2005)", fontsize=11, color=MUTED)
        _bullets(ax, 0.7, 3.9, [
            "Pressure peaks at the edge, temperature behind it on the rake face (crater zone).",
            "Sliding speed: zero at the stagnation point, chip speed on the rake, cutting speed on the flank.",
            "Flank contact is geometric: nodes within an elastic-recovery gap of the lowest point rub - so the wear land grows with wear.",
            "Demo constants are illustrative; the time step is auto-scaled (numerical control, not calibration).",
        ], size=11.5, gap=0.5, width_chars=50)
        _image(fig, images / "fig3_process_fields.png", (6.9, 0.55, 5.9, 5.6))
        pdf.savefig(fig)

        # 5 - geometry update ---------------------------------------------------
        fig, ax = _slide(pdf, 5, "Geometry update that keeps the mesh healthy", "numerics")
        _image(fig, images / "cycle_001_geometry_after.png", (0.5, 1.75, 6.2, 4.4))
        _image(fig, images / "fig6_displacement_heatmap.png", (6.7, 1.75, 6.2, 4.4))
        ax.text(0.7, 1.35, "Wear depth x direction (flank: normal to cutting velocity; rake: surface normal) -> nodes slide back to their original spacing along the worn profile",
                fontsize=10.5, color=INK)
        ax.text(0.7, 0.95, "-> interior follows via normalised, tapered inverse-distance weights (|u_interior| <= max |u_surface|) -> scaled-Jacobian check, step x 0.5 on failure",
                fontsize=10.5, color=INK)
        pdf.savefig(fig)

        # 6 - results ---------------------------------------------------------
        fig, ax = _slide(pdf, 6, "Multi-cycle result: crater and flank wear land emerge", "results")
        _image(fig, images / "fig4_wear_progression.png", (0.5, 0.95, 6.0, 5.2))
        _image(fig, images / "fig5_wear_indicator.png", (6.6, 2.0, 6.3, 3.6))
        if stats:
            ax.text(6.8, 1.55, f"{stats['cycles']} cycles  |  {stats['time']:.0f} min equivalent cutting time (synthetic)", fontsize=11, color=INK)
            ax.text(6.8, 1.15, f"flank-wear indicator {stats['vb']:.0f} um  |  crater depth {stats['crater']:.0f} um", fontsize=11, color=INK)
            ax.text(6.8, 0.75, f"stopped by the flank-wear criterion; no inverted elements (min scaled Jacobian {stats['q']:.2f})", fontsize=11, color=INK)
        pdf.savefig(fig)

        # 7 - value -----------------------------------------------------------
        fig, ax = _slide(pdf, 7, "Engineering value and skills demonstrated", "takeaways")
        cols = [
            ("Python & automation", ["modular package, CLI, config-driven runs", "checkpoint / resume, per-cycle outputs"]),
            ("FEM workflow thinking", ["solver -> fields -> wear -> geometry -> solver", "backend interface ready for a solver adapter"]),
            ("Manufacturing physics", ["orthogonal cutting, Usui wear law", "crater vs. flank wear, VB-type indicator"]),
            ("Numerical geometry", ["contour normals, node relocation", "bounded smoothing, Jacobian-based quality"]),
            ("Reproducibility & testing", ["seeded synthetic data, 74 tests", "GitHub Actions CI on 3.11 / 3.12"]),
            ("Communication", ["figures, GIF, HTML dashboard", "methodology, validation and limits documented"]),
        ]
        for i, (head, items) in enumerate(cols):
            x = 0.7 + (i % 3) * 4.1
            y = 4.25 - (i // 3) * 2.05
            ax.add_patch(FancyBboxPatch((x, y), 3.8, 1.7, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#F6F7F9", ec="#E5E7EB"))
            ax.text(x + 0.25, y + 1.3, head, fontsize=14, fontweight="bold", color=NAVY)
            for k, it in enumerate(items):
                ax.text(x + 0.25, y + 0.8 - k * 0.42, "- " + it, fontsize=10.5, color=INK)
        ax.text(0.7, 0.95, "Independently developed demonstrator inspired by published literature. Not an official RWTH Aachen / MTI repository; "
                           "contains no confidential data or code.", fontsize=10, color=MUTED, style="italic")
        pdf.savefig(fig)

        info = pdf.infodict()
        info["Title"] = "Synthetic Multi-Cycle Tool-Wear Simulation Framework - Project Overview"
        info["Author"] = "Satagni Ray"
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    build(ROOT / "docs" / "project_overview.pdf", ROOT / "docs" / "images")
