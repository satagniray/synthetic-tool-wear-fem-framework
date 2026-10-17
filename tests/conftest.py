from __future__ import annotations

import pytest

from tool_wear_demo.core.contour import ToolContour
from tool_wear_demo.core.mesh import MeshSpec, ToolGeometry, create_tool_mesh
from tool_wear_demo.io.config import load_config

SMALL_MESH = {"n_flank": 12, "n_edge": 6, "n_rake": 14, "n_depth": 10, "surface_growth": 1.15, "depth_exponent": 1.5}


@pytest.fixture
def small_mesh():
    return create_tool_mesh(ToolGeometry(), MeshSpec(**SMALL_MESH))


@pytest.fixture
def contour(small_mesh):
    return ToolContour(small_mesh)


@pytest.fixture
def small_config(tmp_path):
    return load_config(
        None,
        {
            "mesh": SMALL_MESH,
            "wear": {"max_cycles": 3, "target_flank_wear_mm": 10.0},
            "output": {"directory": str(tmp_path / "results"), "save_png": False, "save_html": False},
        },
    )
