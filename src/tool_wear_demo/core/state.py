"""Plain data containers passed between the workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProcessFieldState:
    """Local process variables on the wear-contour nodes for one cycle.

    In the synthetic backend these are analytical stand-ins; a solver backend
    would fill the same container with values extracted from FEM results.
    All arrays are ordered like :attr:`ToolContour.node_index`.
    """

    node_index: np.ndarray  # zero-based mesh node indices
    surface_type: np.ndarray
    s_norm: np.ndarray  # normalised surface coordinate [0, 1]
    d_mm: np.ndarray  # signed distance from the edge middle (flank < 0 < rake)
    x_mm: np.ndarray
    y_mm: np.ndarray
    pressure_MPa: np.ndarray
    temperature_K: np.ndarray
    sliding_velocity_m_s: np.ndarray
    source: str = "synthetic"

    @property
    def in_contact(self) -> np.ndarray:
        return self.pressure_MPa > 0.0

    def __len__(self) -> int:
        return int(self.node_index.size)

    def as_columns(self) -> dict[str, np.ndarray]:
        return {
            "node_id": self.node_index + 1,
            "surface_type": self.surface_type,
            "s_norm": self.s_norm,
            "d_mm": self.d_mm,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "contact_pressure_MPa": self.pressure_MPa,
            "temperature_K": self.temperature_K,
            "sliding_velocity_m_s": self.sliding_velocity_m_s,
            "in_contact": self.in_contact.astype(int),
        }


@dataclass
class SimulationState:
    """Progress of a multi-cycle run (what a checkpoint must restore)."""

    cycle: int = 0
    elapsed_time_s: float = 0.0
    coords: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)
