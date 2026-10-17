"""Backend interfaces: where the process fields of a cycle come from.

The wear loop only talks to these protocols, so the source of the local
contact pressure / temperature / sliding velocity can be swapped:

* :class:`~tool_wear_demo.backends.synthetic.SyntheticBackend` (default,
  solver-free, analytical fields);
* a solver adapter (see ``abaqus_stub.py``) that would write a model for the
  current geometry, run a coupled cutting simulation and extract the same
  three fields on the tool surface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..core.mesh import Mesh
from ..core.state import ProcessFieldState


@runtime_checkable
class ProcessFieldProvider(Protocol):
    """Access to the local process variables of the last solved cycle."""

    def get_contact_pressure(self, state: ProcessFieldState) -> np.ndarray: ...

    def get_temperature(self, state: ProcessFieldState) -> np.ndarray: ...

    def get_sliding_velocity(self, state: ProcessFieldState) -> np.ndarray: ...


@runtime_checkable
class SolverBackend(Protocol):
    """Runs one process "simulation" for the current tool geometry."""

    name: str

    def run_cycle(self, mesh: Mesh, cycle: int) -> ProcessFieldState: ...
