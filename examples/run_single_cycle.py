"""Run ONE wear cycle on the synthetic tool and write its output folder.

    python examples/run_single_cycle.py

Shows the single-cycle pipeline step by step:
synthetic fields -> wear rate -> wear depth -> surface motion -> interior
smoothing -> mesh-quality check -> metrics -> CSV/JSON/PNG output.
"""

from __future__ import annotations

from pathlib import Path

from tool_wear_demo import build_cycle_runner, load_config, mesh_from_config
from tool_wear_demo.io.results import RunDirectory

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "config" / "demo_fast.yaml")
    mesh = mesh_from_config(config)
    runner = build_cycle_runner(config, mesh)

    run = RunDirectory.create(ROOT / "results", name="single_cycle_demo")
    result = runner.run(mesh, cycle_index=1, cycle_dir=run.cycle_dir(1))

    print(f"mesh:                       {mesh.n_nodes} nodes, {mesh.n_elements} quads")
    print(f"wear-contour nodes:         {len(runner.contour)}")
    print(f"cycle duration:             {result.delta_t_s:.1f} s (auto-scaled, synthetic)")
    print(f"max surface move:           {result.max_surface_move_um:.2f} um")
    print(f"mesh quality:               valid={result.quality.valid}, min scaled Jacobian={result.quality.min_quality:.3f}, retries={result.quality.retries}")
    for key, value in result.metrics.items():
        print(f"{key + ':':28s}{value:10.3f}")
    print(f"outputs written to:         {run.cycle_dir(1)}")


if __name__ == "__main__":
    main()
