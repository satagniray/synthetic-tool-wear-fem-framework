"""Optional Abaqus backend - intentionally a stub in the public demonstrator.

The public repository runs entirely without a commercial solver. This module
only documents the shape a solver adapter would take; it contains no input
deck, no solver-specific model data and no result files.

A solver adapter implementing :class:`~tool_wear_demo.backends.base.SolverBackend`
would, for each cycle:

1. write a *newly created, generic* orthogonal-cutting input model whose tool
   nodes take the coordinates of the current :class:`~tool_wear_demo.core.mesh.Mesh`;
2. submit the coupled thermo-mechanical cutting analysis;
3. read standard, documented surface outputs on the tool (contact pressure,
   nodal temperature, relative sliding velocity) through the solver's public
   results API;
4. map them onto the wear-contour nodes and return a
   :class:`~tool_wear_demo.core.state.ProcessFieldState`.

Everything downstream (wear law, directions, smoothing, quality control,
checkpointing) is shared with the synthetic backend.
"""

from __future__ import annotations

from ..core.mesh import Mesh
from ..core.state import ProcessFieldState


class AbaqusBackend:
    """Placeholder for an optional solver adapter (not part of the public demo)."""

    name = "abaqus"

    def __init__(self, *args, **kwargs) -> None:
        self.options = kwargs

    def run_cycle(self, mesh: Mesh, cycle: int) -> ProcessFieldState:
        raise NotImplementedError("Optional Abaqus integration is intentionally excluded from the public demo.")
