import json

import numpy as np

from tool_wear_demo.core.quality import MeshQualityChecker
from tool_wear_demo.workflow.wear_loop import build_cycle_runner

EXPECTED_CSV = ["nodes_before.csv", "nodes_after.csv", "process_fields.csv", "wear_rates.csv", "displacement.csv"]
EXPECTED_PNG = ["geometry_before.png", "geometry_after.png", "process_fields.png"]


def test_single_cycle_produces_wear(small_mesh, small_config):
    runner = build_cycle_runner(small_config, small_mesh)
    result = runner.run(small_mesh, 1)
    assert result.quality.valid
    assert not np.allclose(result.updated_mesh.coords, small_mesh.coords)
    target = small_config["wear"]["target_max_surface_displacement_um"]
    assert 0 < result.max_surface_move_um <= target * 1.05
    assert result.metrics["max_recession_um"] > 0
    assert result.delta_t_s > 0 and result.elapsed_time_s == result.delta_t_s
    # back faces untouched
    fixed = small_mesh.fixed_mask
    assert np.array_equal(result.updated_mesh.coords[fixed], small_mesh.coords[fixed])
    # tool loses material
    assert result.updated_mesh.area() < small_mesh.area()


def test_single_cycle_output_folder(tmp_path, small_mesh, small_config):
    config = dict(small_config)
    config["output"] = dict(config["output"], save_png=True)
    runner = build_cycle_runner(config, small_mesh)
    cycle_dir = tmp_path / "cycle_001"
    runner.run(small_mesh, 1, cycle_dir=cycle_dir)
    for name in EXPECTED_CSV + EXPECTED_PNG + ["mesh_quality.json", "summary.json"]:
        assert (cycle_dir / name).exists(), name
    quality = json.loads((cycle_dir / "mesh_quality.json").read_text())
    assert quality["valid"] is True and quality["retries"] == 0
    summary = json.loads((cycle_dir / "summary.json").read_text())
    assert summary["cycle"] == 1 and summary["status"] == "done" and summary["data_source"] == "synthetic"


def test_single_cycle_is_deterministic(small_mesh, small_config):
    a = build_cycle_runner(small_config, small_mesh).run(small_mesh, 1)
    b = build_cycle_runner(small_config, small_mesh).run(small_mesh, 1)
    assert np.array_equal(a.updated_mesh.coords, b.updated_mesh.coords)


def test_updated_mesh_passes_quality(small_mesh, small_config):
    result = build_cycle_runner(small_config, small_mesh).run(small_mesh, 1)
    assert MeshQualityChecker(small_config["quality"]["min_scaled_jacobian"]).check(result.updated_mesh.coords, small_mesh.elements).valid
