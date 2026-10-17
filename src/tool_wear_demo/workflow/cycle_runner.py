"""One wear cycle: fields -> wear -> surface motion -> smoothing -> quality -> metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..backends.base import SolverBackend
from ..core.contour import ToolContour
from ..core.mesh import Mesh
from ..core.quality import MeshQualityChecker, QualityReport
from ..core.state import ProcessFieldState
from ..displacement.interior_smoothing import InteriorSmoother
from ..displacement.surface_update import redistribute_surface_nodes, surface_displacement
from ..io import csv_io
from ..io.results import write_json
from ..physics.wear_models import WearModel, smooth_along_contour, time_step
from .metrics import compute_wear_metrics


class CycleFailedError(RuntimeError):
    """Raised when no displacement scale yields an acceptable mesh."""

    def __init__(self, message: str, report: QualityReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class CycleResult:
    cycle: int
    mesh_before: Mesh
    updated_mesh: Mesh
    fields: ProcessFieldState
    wear_rate_mm_s: np.ndarray
    wear_depth_mm: np.ndarray  # applied (filtered, scaled) depth per contour node
    surface_displacement_mm: np.ndarray  # final contour-node displacement (after relocation)
    displacement_mm: np.ndarray  # full nodal displacement field
    quality: QualityReport
    delta_t_s: float
    elapsed_time_s: float
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def flank_wear(self) -> float:
        return self.metrics["vb_indicator_mm"]

    @property
    def max_surface_move_um(self) -> float:
        return float(np.linalg.norm(self.surface_displacement_mm, axis=1).max() * 1000.0)

    def summary(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "status": "done",
            "elapsed_time_s": self.elapsed_time_s,
            "delta_t_s": self.delta_t_s,
            "max_surface_move_um": self.max_surface_move_um,
            "displacement_scale": self.quality.displacement_scale,
            "retries": self.quality.retries,
            **self.metrics,
            "min_scaled_jacobian": self.quality.min_quality,
            "data_source": self.fields.source,
        }


class CycleRunner:
    """Executes a single wear cycle on a mesh and optionally writes its outputs."""

    def __init__(
        self,
        config: Mapping[str, Any],
        backend: SolverBackend,
        contour: ToolContour,
        wear_model: WearModel,
        smoother: InteriorSmoother,
        checker: MeshQualityChecker,
        reference_mesh: Mesh,
    ) -> None:
        self.config = config
        self.backend = backend
        self.contour = contour
        self.wear_model = wear_model
        self.smoother = smoother
        self.checker = checker
        self.reference_mesh = reference_mesh
        self.reference_fractions = contour.anchored_fractions(reference_mesh.coords)

    def run(self, mesh: Mesh, cycle_index: int, elapsed_time_s: float = 0.0, cycle_dir: Path | None = None) -> CycleResult:
        wear_cfg = self.config["wear"]
        quality_cfg = self.config["quality"]
        coords = mesh.coords

        # 1-2. process fields on the active (contour) nodes
        fields = self.backend.run_cycle(mesh, cycle_index)
        # 3. local wear rate
        rate = self.wear_model.wear_rate(fields.pressure_MPa, fields.temperature_K, fields.sliding_velocity_m_s)
        # 4. wear depth for this cycle
        dt = time_step(rate, wear_cfg["time_step_mode"], wear_cfg["target_max_surface_displacement_um"] / 1000.0, wear_cfg["delta_t_s"])
        depth = smooth_along_contour(rate * dt, int(wear_cfg["depth_smoothing_passes"]))

        scale = 1.0
        retries = 0
        report: QualityReport | None = None
        for _ in range(int(quality_cfg["max_retries"]) + 1):
            # 5. surface displacement vectors (+ tangential node relocation)
            surf = surface_displacement(self.contour, coords, depth, wear_cfg["flank_direction"], scale)
            if self.config["smoothing"]["redistribute_surface_nodes"]:
                surf = redistribute_surface_nodes(self.contour, coords, surf, self.reference_fractions)
            # 6. smooth into the interior
            full = self.smoother.smooth(mesh, self.contour.node_index, surf)
            # 7-8. trial geometry + quality check
            trial = coords + full
            report = self.checker.check(trial, mesh.elements)
            if report.valid:
                break
            # 9. reduce and retry
            scale *= float(quality_cfg["retry_scale"])
            retries += 1
        else:
            assert report is not None
            report.retries = retries
            report.displacement_scale = scale
            raise CycleFailedError(
                f"cycle {cycle_index}: mesh quality not acceptable after {retries} retries "
                f"(min scaled Jacobian {report.min_quality:.3f})",
                report,
            )

        report.retries = retries
        report.displacement_scale = scale
        new_mesh = mesh.with_coords(trial)
        dt_eff = dt * scale
        # 10. metrics
        metrics = compute_wear_metrics(self.reference_mesh, new_mesh, self.contour, wear_cfg["vb_threshold_um"])
        result = CycleResult(
            cycle=int(cycle_index),
            mesh_before=mesh,
            updated_mesh=new_mesh,
            fields=fields,
            wear_rate_mm_s=rate,
            wear_depth_mm=depth * scale,
            surface_displacement_mm=surf,
            displacement_mm=full,
            quality=report,
            delta_t_s=dt_eff,
            elapsed_time_s=elapsed_time_s + dt_eff,
            metrics=metrics,
        )
        # 11. outputs
        if cycle_dir is not None:
            save_cycle_outputs(Path(cycle_dir), result, self.contour, self.config)
        return result


def save_cycle_outputs(cycle_dir: Path, result: CycleResult, contour: ToolContour, config: Mapping[str, Any]) -> None:
    """Write the per-cycle CSV, JSON and (optionally) PNG files."""
    out = config["output"]
    cycle_dir.mkdir(parents=True, exist_ok=True)
    mesh = result.mesh_before
    if out["save_csv"]:
        csv_io.write_nodes(cycle_dir / "nodes_before.csv", mesh)
        csv_io.write_nodes(cycle_dir / "nodes_after.csv", result.updated_mesh)
        csv_io.write_table(cycle_dir / "process_fields.csv", result.fields.as_columns())
        csv_io.write_table(
            cycle_dir / "wear_rates.csv",
            {
                "node_id": contour.node_index + 1,
                "surface_type": contour.surface_type,
                "d_mm": result.fields.d_mm,
                "wear_rate_mm_s": result.wear_rate_mm_s,
                "wear_depth_um": result.wear_depth_mm * 1000.0,
            },
        )
        disp = result.displacement_mm * 1000.0
        kind = np.where(mesh.fixed_mask, "fixed", "interior")
        kind[contour.node_index] = "surface"
        csv_io.write_table(
            cycle_dir / "displacement.csv",
            {"node_id": mesh.node_ids, "ux_um": disp[:, 0], "uy_um": disp[:, 1], "magnitude_um": np.linalg.norm(disp, axis=1), "kind": kind},
        )
    write_json(cycle_dir / "mesh_quality.json", result.quality.to_dict())
    write_json(cycle_dir / "summary.json", result.summary())
    if out["save_png"]:
        from ..visualization.mesh_plot import plot_cycle_geometry
        from ..visualization.wear_plot import plot_process_fields

        zoom = float(out["zoom_window_mm"])
        plot_cycle_geometry(mesh, contour, cycle_dir / "geometry_before.png", zoom, title=f"Cycle {result.cycle:03d} - before update")
        plot_cycle_geometry(
            result.updated_mesh,
            contour,
            cycle_dir / "geometry_after.png",
            zoom,
            title=f"Cycle {result.cycle:03d} - after update",
            previous_coords=mesh.coords,
            surface_displacement=result.surface_displacement_mm,
        )
        plot_process_fields(result.fields, cycle_dir / "process_fields.png", wear_rate=result.wear_rate_mm_s, title=f"Synthetic process fields - cycle {result.cycle:03d}")
