"""Run-folder layout and result writers.

::

    results/
      LATEST                          <- name of the most recent run folder
      run_YYYYMMDD_HHMMSS/
        config_used.yaml              frozen configuration of the run
        nodes_initial.csv, elements.csv
        history.csv                   one row per cycle (cycle 0 = initial state)
        checkpoint.json, state.npz    restart information
        figures/                      run-level plots, GIF, dashboard.html
        cycle_001/ ...                per-cycle CSV / JSON / PNG output
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

LATEST_POINTER = "LATEST"


class RunDirectory:
    """Paths of one simulation run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def create(cls, base: str | Path, name: str | None = None) -> "RunDirectory":
        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        stem = name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        candidate = base / stem
        k = 1
        while candidate.exists():
            candidate = base / f"{stem}_{k}"
            k += 1
        candidate.mkdir(parents=True)
        (base / LATEST_POINTER).write_text(candidate.name + "\n", encoding="utf-8")
        return cls(candidate)

    # well-known files
    @property
    def config_file(self) -> Path:
        return self.path / "config_used.yaml"

    @property
    def checkpoint_file(self) -> Path:
        return self.path / "checkpoint.json"

    @property
    def state_file(self) -> Path:
        return self.path / "state.npz"

    @property
    def history_file(self) -> Path:
        return self.path / "history.csv"

    @property
    def nodes_initial_file(self) -> Path:
        return self.path / "nodes_initial.csv"

    @property
    def elements_file(self) -> Path:
        return self.path / "elements.csv"

    @property
    def figures(self) -> Path:
        p = self.path / "figures"
        p.mkdir(exist_ok=True)
        return p

    def cycle_dir(self, cycle: int, create: bool = True) -> Path:
        p = self.path / f"cycle_{int(cycle):03d}"
        if create:
            p.mkdir(exist_ok=True)
        return p

    def cycle_dirs(self) -> list[Path]:
        return sorted(p for p in self.path.glob("cycle_*") if p.is_dir())


def resolve_run_dir(path: str | Path) -> Path:
    """Accept a run folder, or ``<results>/latest`` (resolved via the LATEST pointer)."""
    p = Path(path)
    if p.is_dir() and (p / "config_used.yaml").exists():
        return p
    if p.name.lower() == "latest":
        pointer = p.parent / LATEST_POINTER
        if pointer.exists():
            target = p.parent / pointer.read_text(encoding="utf-8").strip()
            if target.is_dir():
                return target
    raise FileNotFoundError(f"no simulation run found at '{path}'")


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2)
        handle.write("\n")


def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
