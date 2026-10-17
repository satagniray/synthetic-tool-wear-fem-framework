"""Synthetic multi-cycle tool-wear simulation framework.

A solver-free portfolio demonstrator of an automated, FEM-style tool-wear loop
for 2D orthogonal cutting. All geometry, process fields and results are
synthetic and not experimentally validated.
"""

__version__ = "1.0.0"

from .core.contour import ToolContour
from .core.mesh import Mesh, MeshSpec, ToolGeometry, create_tool_mesh, mesh_from_config
from .core.quality import MeshQualityChecker
from .io.config import default_config, load_config
from .workflow.cycle_runner import CycleRunner
from .workflow.wear_loop import WearLoop, build_cycle_runner, run_wear_loop

__all__ = [
    "__version__",
    "ToolContour",
    "Mesh",
    "MeshSpec",
    "ToolGeometry",
    "create_tool_mesh",
    "mesh_from_config",
    "MeshQualityChecker",
    "default_config",
    "load_config",
    "CycleRunner",
    "WearLoop",
    "build_cycle_runner",
    "run_wear_loop",
]
