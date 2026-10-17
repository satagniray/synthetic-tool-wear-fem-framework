"""Geometry, mesh, contour, quality and state primitives."""

from .contour import ToolContour, boundary_loop
from .mesh import Mesh, MeshSpec, ToolGeometry, create_tool_mesh, mesh_from_config
from .quality import MeshQualityChecker, QualityReport, signed_areas
from .state import ProcessFieldState, SimulationState

__all__ = [
    "Mesh",
    "MeshSpec",
    "ToolGeometry",
    "create_tool_mesh",
    "mesh_from_config",
    "ToolContour",
    "boundary_loop",
    "MeshQualityChecker",
    "QualityReport",
    "signed_areas",
    "ProcessFieldState",
    "SimulationState",
]
