"""Solver-free backend producing synthetic process fields."""

from __future__ import annotations

import numpy as np

from ..core.contour import ToolContour
from ..core.mesh import Mesh
from ..core.state import ProcessFieldState
from ..physics.synthetic_fields import SyntheticFieldParameters, generate_process_fields


class SyntheticBackend:
    """Implements :class:`SolverBackend` and :class:`ProcessFieldProvider` analytically.

    The random generator is seeded with ``(seed, cycle)`` so every cycle is
    reproducible on its own - a resumed run gives exactly the same result as an
    uninterrupted one.
    """

    name = "synthetic"

    def __init__(self, contour: ToolContour, params: SyntheticFieldParameters | None = None, seed: int = 42) -> None:
        self.contour = contour
        self.params = params or SyntheticFieldParameters()
        self.seed = int(seed)

    def run_cycle(self, mesh: Mesh, cycle: int) -> ProcessFieldState:
        rng = np.random.default_rng([self.seed, int(cycle)])
        return generate_process_fields(self.contour, mesh.coords, self.params, rng)

    # ProcessFieldProvider -------------------------------------------------
    def get_contact_pressure(self, state: ProcessFieldState) -> np.ndarray:
        return state.pressure_MPa

    def get_temperature(self, state: ProcessFieldState) -> np.ndarray:
        return state.temperature_K

    def get_sliding_velocity(self, state: ProcessFieldState) -> np.ndarray:
        return state.sliding_velocity_m_s
