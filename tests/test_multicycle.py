import json

import numpy as np
import pytest

from tool_wear_demo.cli import main
from tool_wear_demo.core.quality import signed_areas
from tool_wear_demo.io import csv_io
from tool_wear_demo.io.config import ConfigError, load_config
from tool_wear_demo.io.results import resolve_run_dir
from tool_wear_demo.workflow.wear_loop import WearLoop


def test_three_cycle_integration(tmp_path, small_config):
    loop = WearLoop(small_config, tmp_path / "run")
    result = loop.run(progress=None)
    run = tmp_path / "run"

    # output folders exist
    for c in (1, 2, 3):
        assert (run / f"cycle_{c:03d}" / "summary.json").exists()
    for name in ("config_used.yaml", "nodes_initial.csv", "elements.csv", "history.csv", "checkpoint.json", "state.npz"):
        assert (run / name).exists(), name

    # wear increases monotonically
    vb = [h["vb_indicator_mm"] for h in result.history]
    rec = [h["max_recession_um"] for h in result.history]
    area = [h["removed_area_um2"] for h in result.history]
    assert all(b >= a for a, b in zip(vb, vb[1:]))
    assert all(b > a for a, b in zip(rec, rec[1:]))
    assert all(b > a for a, b in zip(area, area[1:]))

    # no element inversion
    assert np.all(signed_areas(result.final_mesh.coords, result.final_mesh.elements) > 0)

    # checkpoint written
    ckpt = json.loads((run / "checkpoint.json").read_text())
    assert ckpt["cycle"] == 3 and ckpt["status"] == "finished" and ckpt["stop_reason"] == "max_cycles_reached"

    # final mesh differs from initial mesh
    assert not np.allclose(result.final_mesh.coords, loop.initial_mesh.coords)
    assert result.cycles_completed == 3


def test_stops_at_flank_wear_target(tmp_path, small_config):
    config = dict(small_config)
    config["wear"] = dict(config["wear"], target_flank_wear_mm=0.02, max_cycles=10)
    result = WearLoop(config, tmp_path / "run").run(progress=None)
    assert result.stop_reason == "target_flank_wear_reached"
    assert result.final_flank_wear_mm >= 0.02
    assert result.cycles_completed < 10


def test_resume_reproduces_uninterrupted_run(tmp_path, small_config):
    full = WearLoop(small_config, tmp_path / "full").run(progress=None)

    first = dict(small_config)
    first["wear"] = dict(first["wear"], max_cycles=1)
    WearLoop(first, tmp_path / "split").run(progress=None)
    resumed = WearLoop.resume(tmp_path / "split")
    resumed.config["wear"]["max_cycles"] = 3
    result = resumed.run(progress=None)

    assert result.cycles_completed == 3
    assert np.allclose(result.final_mesh.coords, full.final_mesh.coords, atol=1e-12)
    hist = csv_io.read_rows(tmp_path / "split" / "history.csv")
    assert [int(h["cycle"]) for h in hist] == [0, 1, 2, 3]


def test_in_memory_run_writes_nothing(tmp_path, small_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = WearLoop(small_config, run_dir=False).run(progress=None)
    assert result.run_dir is None and result.cycles_completed == 3
    assert not (tmp_path / "results").exists()


def test_latest_pointer_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "results"
    assert main(["run", "--max-cycles", "2", "--output", str(out), "--no-png"]) == 0
    run_dir = resolve_run_dir(out / "latest")
    assert (run_dir / "cycle_002").exists()
    assert main(["generate-demo-data", "--out", str(tmp_path / "data")]) == 0
    assert (tmp_path / "data" / "example_process_fields.csv").exists()
    assert (tmp_path / "data" / "example_process_fields.meta.yaml").exists()
    assert main(["validate"]) == 0
    assert "All checks passed" in capsys.readouterr().out


def test_figures_and_dashboard(tmp_path, small_config):
    config = dict(small_config)
    config["output"] = dict(config["output"], save_png=True, save_html=True)
    config["wear"] = dict(config["wear"], max_cycles=2)
    WearLoop(config, tmp_path / "run").run(progress=None)
    figs = tmp_path / "run" / "figures"
    for name in (
        "fig1_workflow.png", "fig2_initial_mesh.png", "fig3_process_fields.png", "fig4_wear_progression.png",
        "fig5_wear_indicator.png", "fig6_displacement_heatmap.png", "wear_progression.gif", "dashboard.html",
    ):
        assert (figs / name).exists(), name
    html = (figs / "dashboard.html").read_text()
    assert "__DATA__" not in html and '"cycles"' in html


def test_config_validation(tmp_path):
    with pytest.raises(ConfigError):
        load_config(None, {"wear": {"model": "unknown"}})
    with pytest.raises(ConfigError):
        load_config(None, {"wear": {"typo_key": 1}})
    with pytest.raises(ConfigError):
        load_config(None, {"quality": {"retry_scale": 1.5}})
    cfg = tmp_path / "c.yaml"
    cfg.write_text("wear:\n  max_cycles: 4\n")
    assert load_config(cfg)["wear"]["max_cycles"] == 4
