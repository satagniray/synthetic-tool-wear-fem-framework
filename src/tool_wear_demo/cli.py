"""Command-line interface.

Examples::

    python -m tool_wear_demo.cli run --config config/default.yaml
    python -m tool_wear_demo.cli run --config config/demo_fast.yaml
    python -m tool_wear_demo.cli run --resume results/run_20261011_120000
    python -m tool_wear_demo.cli plot results/latest
    python -m tool_wear_demo.cli validate
    python -m tool_wear_demo.cli generate-demo-data
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


def _cmd_run(args: argparse.Namespace) -> int:
    from .io.config import load_config
    from .workflow.wear_loop import WearLoop

    if args.resume:
        loop = WearLoop.resume(args.resume)
        if args.max_cycles:
            loop.config["wear"]["max_cycles"] = int(args.max_cycles)
        print(f"Resuming {loop.run_dir} from cycle {loop._start_cycle}")
    else:
        overrides: dict = {}
        if args.max_cycles:
            overrides.setdefault("wear", {})["max_cycles"] = int(args.max_cycles)
        if args.output:
            overrides.setdefault("output", {})["directory"] = str(args.output)
        if args.no_png:
            overrides.setdefault("output", {}).update(save_png=False, save_html=False)
        config = load_config(args.config, overrides)
        loop = WearLoop(config)
        print(f"Run folder: {loop.run_dir}")
    result = loop.run()
    print(f"Final flank-wear indicator: {result.final_flank_wear_mm * 1000:.1f} um ({result.stop_reason})")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from .visualization.dashboard import make_run_outputs

    paths = make_run_outputs(args.run_dir)
    for name, path in paths.items():
        print(f"{name:22s} {path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Self-check: configuration, geometry, physics sanity and a short solver-free run."""
    from .core.contour import ToolContour, points_in_polygon
    from .core.mesh import mesh_from_config
    from .core.quality import MeshQualityChecker
    from .io.config import load_config
    from .physics.wear_models import UsuiPublishedParameters, build_wear_model
    from .workflow.wear_loop import WearLoop

    checks: list[tuple[str, bool, str]] = []
    config = load_config(args.config)
    checks.append(("configuration loads and validates", True, args.config or "built-in defaults"))

    mesh = mesh_from_config(config)
    q = MeshQualityChecker(config["quality"]["min_scaled_jacobian"]).check(mesh.coords, mesh.elements)
    checks.append(("initial mesh is valid", q.valid, f"{mesh.n_elements} quads, min scaled Jacobian {q.min_quality:.3f}"))

    contour = ToolContour(mesh)
    order = list(dict.fromkeys(contour.surface_type.tolist()))
    checks.append(("contour ordered flank -> edge -> rake", order == ["flank", "edge", "rake"], " -> ".join(order)))
    n = contour.inward_normals(mesh.coords)
    inside = points_in_polygon(contour.points(mesh.coords) + 1e-4 * n, contour.boundary_polygon(mesh.coords))
    checks.append(("inward normals point into the tool", bool(inside.all()), f"{int(inside.sum())}/{inside.size}"))

    model = build_wear_model(config["wear"]["model"])
    r = model.wear_rate([0.0, 1000.0, 1000.0, 1000.0], [900.0, 900.0, 900.0, 1000.0], [2.0, 0.0, 2.0, 2.0])
    checks.append(("wear law: zero pressure / zero sliding -> zero wear", r[0] == 0 and r[1] == 0, ""))
    checks.append(("wear law: hotter -> faster", r[3] > r[2] > 0, f"{r[2]:.3e} < {r[3]:.3e} mm/s"))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(args.config, {"wear": {"max_cycles": 2}, "output": {"save_png": False, "save_html": False}})
        result = WearLoop(cfg, Path(tmp) / "validate_run").run(progress=None)
        vb = [h["vb_indicator_mm"] for h in result.history]
        checks.append(("2-cycle smoke run completes", result.cycles_completed == 2, f"stop: {result.stop_reason}"))
        checks.append(("wear indicator does not decrease", all(b >= a for a, b in zip(vb, vb[1:])), f"{[round(v * 1000, 1) for v in vb]} um"))

    width = max(len(c[0]) for c in checks)
    ok = True
    for name, passed, info in checks:
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name:<{width}}  {info}")
    if not UsuiPublishedParameters().verified_for_quantitative_use:
        print("[NOTE] 'usui_published' constants are literature reference values; verify units before quantitative use.")
    print("All checks passed." if ok else "Some checks FAILED.")
    return 0 if ok else 1


def _cmd_generate(args: argparse.Namespace) -> int:
    """Write example synthetic process fields for the unworn tool (cycle 1)."""
    from .backends.synthetic import SyntheticBackend
    from .core.contour import ToolContour
    from .core.mesh import mesh_from_config
    from .io.config import load_config
    from .io.csv_io import write_table
    from .physics.synthetic_fields import SyntheticFieldParameters
    from .physics.wear_models import build_wear_model

    config = load_config(args.config)
    mesh = mesh_from_config(config)
    contour = ToolContour(mesh)
    seed = int(config["project"]["seed"])
    backend = SyntheticBackend(contour, SyntheticFieldParameters.from_mapping(config["synthetic_fields"]), seed)
    fields = backend.run_cycle(mesh, 1)
    rate = build_wear_model(config["wear"]["model"]).wear_rate(fields.pressure_MPa, fields.temperature_K, fields.sliding_velocity_m_s)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = fields.as_columns()
    columns["wear_rate_um_per_min"] = np.asarray(rate) * 6.0e4
    csv_path = out / "example_process_fields.csv"
    write_table(csv_path, columns)
    meta = {
        "source": "synthetic",
        "seed": seed,
        "purpose": "portfolio demonstration",
        "not_for_physical_validation": True,
        "generated_by": "python -m tool_wear_demo.cli generate-demo-data",
        "config": args.config or "built-in defaults",
        "units": {
            "s_norm": "-",
            "d_mm": "mm (signed distance from edge middle, flank < 0 < rake)",
            "x_mm": "mm",
            "y_mm": "mm",
            "contact_pressure_MPa": "MPa",
            "temperature_K": "K",
            "sliding_velocity_m_s": "m/s",
            "wear_rate_um_per_min": "um/min (wear model: " + config["wear"]["model"] + ")",
        },
    }
    with open(out / "example_process_fields.meta.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False)
    print(f"wrote {csv_path} ({len(fields)} surface nodes) and its metadata")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tool-wear-demo", description="Synthetic multi-cycle tool-wear simulation framework")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the multi-cycle wear loop")
    run.add_argument("--config", default="config/default.yaml" if Path("config/default.yaml").exists() else None)
    run.add_argument("--resume", metavar="RUN_DIR", help="continue an existing run from its checkpoint")
    run.add_argument("--output", help="results base directory (overrides output.directory)")
    run.add_argument("--max-cycles", type=int, help="override wear.max_cycles")
    run.add_argument("--no-png", action="store_true", help="skip PNG/HTML output (fast)")
    run.set_defaults(func=_cmd_run)

    plot = sub.add_parser("plot", help="(re)create figures, GIF and dashboard for a run")
    plot.add_argument("run_dir", help="run folder or <results>/latest")
    plot.set_defaults(func=_cmd_plot)

    val = sub.add_parser("validate", help="self-check configuration, geometry, physics and a 2-cycle run")
    val.add_argument("--config", default=None)
    val.set_defaults(func=_cmd_validate)

    gen = sub.add_parser("generate-demo-data", help="write data/synthetic/example_process_fields.csv")
    gen.add_argument("--config", default=None)
    gen.add_argument("--out", default="data/synthetic")
    gen.set_defaults(func=_cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
