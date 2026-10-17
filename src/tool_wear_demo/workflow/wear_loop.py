"""Multi-cycle wear loop with stopping criteria, checkpointing and resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..backends.synthetic import SyntheticBackend
from ..core.contour import ToolContour
from ..core.mesh import Mesh, mesh_from_config
from ..core.quality import MeshQualityChecker
from ..displacement.interior_smoothing import build_smoother
from ..io import csv_io
from ..io.config import load_config, save_config, validate_config
from ..io.results import RunDirectory, resolve_run_dir
from ..physics.synthetic_fields import SyntheticFieldParameters
from ..physics.wear_models import build_wear_model
from .checkpoints import load_checkpoint, save_checkpoint
from .cycle_runner import CycleFailedError, CycleResult, CycleRunner
from .metrics import compute_wear_metrics

HISTORY_KEYS = (
    "cycle",
    "elapsed_time_s",
    "delta_t_s",
    "max_surface_move_um",
    "vb_indicator_mm",
    "max_recession_um",
    "crater_depth_indicator_um",
    "edge_recession_um",
    "mean_rake_recession_um",
    "mean_flank_recession_um",
    "removed_area_um2",
    "min_scaled_jacobian",
    "retries",
)


@dataclass
class LoopResult:
    run_dir: Path | None
    final_mesh: Mesh
    history: list[dict]
    stop_reason: str
    cycles_completed: int
    results: list[CycleResult] = field(default_factory=list)

    @property
    def final_flank_wear_mm(self) -> float:
        return float(self.history[-1]["vb_indicator_mm"])


def build_cycle_runner(config: Mapping[str, Any], mesh: Mesh) -> CycleRunner:
    """Assemble backend, physics and numerics from a configuration."""
    contour = ToolContour(mesh)
    backend = SyntheticBackend(contour, SyntheticFieldParameters.from_mapping(config["synthetic_fields"]), config["project"]["seed"])
    return CycleRunner(
        config=config,
        backend=backend,
        contour=contour,
        wear_model=build_wear_model(config["wear"]["model"]),
        smoother=build_smoother(config["smoothing"]),
        checker=MeshQualityChecker(config["quality"]["min_scaled_jacobian"]),
        reference_mesh=mesh.copy(),
    )


def _initial_history_row(runner: CycleRunner, mesh: Mesh) -> dict:
    metrics = compute_wear_metrics(mesh, mesh, runner.contour, runner.config["wear"]["vb_threshold_um"])
    q = runner.checker.check(mesh.coords, mesh.elements)
    row = {k: 0.0 for k in HISTORY_KEYS}
    row.update(metrics)
    row.update(cycle=0, retries=0, min_scaled_jacobian=q.min_quality)
    return {k: row[k] for k in HISTORY_KEYS}


class WearLoop:
    """Runs cycles until the flank-wear target or the cycle limit is reached.

    Parameters
    ----------
    config:
        Validated configuration dictionary (see :func:`tool_wear_demo.io.config.load_config`).
    run_dir:
        Output folder. ``None`` creates ``<output.directory>/run_<timestamp>``;
        ``False`` keeps everything in memory (no files written).
    """

    def __init__(self, config: Mapping[str, Any], run_dir: str | Path | None | bool = None) -> None:
        validate_config(config)
        self.config = dict(config)
        self.initial_mesh = mesh_from_config(self.config)
        self.runner = build_cycle_runner(self.config, self.initial_mesh)
        if run_dir is False:
            self.output = None
        elif run_dir is None:
            self.output = RunDirectory.create(self.config["output"]["directory"])
        else:
            path = Path(run_dir)
            path.mkdir(parents=True, exist_ok=True)
            self.output = RunDirectory(path)
        self._start_cycle = 1
        self._mesh = self.initial_mesh
        self._elapsed = 0.0
        self._history: list[dict] = []

    # -- construction from an existing run ---------------------------------
    @classmethod
    def from_config_file(cls, path: str | Path, run_dir: str | Path | None | bool = None) -> "WearLoop":
        return cls(load_config(path), run_dir)

    @classmethod
    def resume(cls, run_dir: str | Path) -> "WearLoop":
        """Continue a run from its last checkpoint (uses the run's frozen configuration)."""
        path = resolve_run_dir(run_dir)
        loop = cls(load_config(path / "config_used.yaml"), path)
        ckpt = load_checkpoint(loop.output)
        loop._mesh = loop.initial_mesh.with_coords(ckpt.coords)
        loop._elapsed = ckpt.elapsed_time_s
        loop._start_cycle = ckpt.cycle + 1
        if loop.output.history_file.exists():
            loop._history = [r for r in csv_io.read_rows(loop.output.history_file) if int(r["cycle"]) <= ckpt.cycle]
        return loop

    @property
    def run_dir(self) -> Path | None:
        return None if self.output is None else self.output.path

    # -- main loop ----------------------------------------------------------
    def run(self, progress: Callable[[str], None] | None = print, make_figures: bool = True) -> LoopResult:
        cfg = self.config
        wear = cfg["wear"]
        max_cycles = int(wear["max_cycles"])
        target = float(wear["target_flank_wear_mm"])
        log = progress or (lambda _msg: None)

        if self.output is not None and self._start_cycle == 1:
            save_config(cfg, self.output.config_file)
            csv_io.write_nodes(self.output.nodes_initial_file, self.initial_mesh)
            csv_io.write_elements(self.output.elements_file, self.initial_mesh)
        if not self._history:
            self._history = [_initial_history_row(self.runner, self.initial_mesh)]

        mesh = self._mesh
        elapsed = self._elapsed
        results: list[CycleResult] = []
        stop_reason = "max_cycles_reached"
        if self._history and float(self._history[-1]["vb_indicator_mm"]) >= target:
            stop_reason = "target_flank_wear_reached"
            max_cycles = self._start_cycle - 1
        cycle = self._start_cycle - 1

        for cycle in range(self._start_cycle, max_cycles + 1):
            cycle_dir = self.output.cycle_dir(cycle) if self.output is not None else None
            try:
                result = self.runner.run(mesh, cycle, elapsed, cycle_dir)
            except CycleFailedError as exc:
                stop_reason = "mesh_quality_limit"
                log(f"[stop] {exc}")
                cycle -= 1
                break
            mesh = result.updated_mesh
            elapsed = result.elapsed_time_s
            results.append(result)
            row = {k: result.summary()[k] for k in HISTORY_KEYS}
            self._history.append(row)
            log(
                f"cycle {cycle:3d} | t = {elapsed / 60:6.1f} min | VB-indicator = {result.flank_wear * 1000:6.1f} um"
                f" | max recession = {result.metrics['max_recession_um']:5.1f} um"
                f" | min quality = {result.quality.min_quality:.3f} | retries = {result.quality.retries}"
            )
            if self.output is not None:
                csv_io.write_rows(self.output.history_file, self._history)
                save_checkpoint(self.output, cycle, mesh.coords, elapsed, result.flank_wear, result.max_surface_move_um)
            if result.flank_wear >= target:
                stop_reason = "target_flank_wear_reached"
                break

        if self.output is not None:
            csv_io.write_rows(self.output.history_file, self._history)
            last = self._history[-1]
            save_checkpoint(
                self.output,
                int(last["cycle"]),
                mesh.coords,
                elapsed,
                float(last["vb_indicator_mm"]),
                float(last["max_surface_move_um"]),
                status="finished",
                stop_reason=stop_reason,
            )
            if make_figures and (cfg["output"]["save_png"] or cfg["output"]["save_html"]):
                from ..visualization.dashboard import make_run_outputs

                make_run_outputs(self.output.path)
        log(f"[done] {stop_reason} after cycle {int(self._history[-1]['cycle'])}" + (f" -> {self.output.path}" if self.output else ""))

        self._mesh, self._elapsed, self._start_cycle = mesh, elapsed, int(self._history[-1]["cycle"]) + 1
        return LoopResult(
            run_dir=self.run_dir,
            final_mesh=mesh,
            history=list(self._history),
            stop_reason=stop_reason,
            cycles_completed=int(self._history[-1]["cycle"]),
            results=results,
        )

    run_loop = run  # alias


def run_wear_loop(config: Mapping[str, Any], run_dir: str | Path | None | bool = None, progress=print) -> LoopResult:
    """Functional entry point: build a :class:`WearLoop` and run it."""
    return WearLoop(config, run_dir).run(progress=progress)
