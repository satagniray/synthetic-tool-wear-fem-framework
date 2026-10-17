"""Checkpoint / resume support.

After each accepted cycle the current nodal coordinates and progress are
saved, so an interrupted run continues exactly where it stopped. Because the
synthetic noise is seeded per cycle, a resumed run reproduces an uninterrupted
run bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.results import RunDirectory, read_json, write_json


@dataclass
class Checkpoint:
    cycle: int
    status: str
    elapsed_time_s: float
    flank_wear_mm: float
    max_surface_move_um: float
    stop_reason: str | None
    coords: np.ndarray


def save_checkpoint(
    run: RunDirectory,
    cycle: int,
    coords: np.ndarray,
    elapsed_time_s: float,
    flank_wear_mm: float,
    max_surface_move_um: float,
    status: str = "done",
    stop_reason: str | None = None,
) -> None:
    np.savez_compressed(run.state_file, coords=coords, cycle=cycle, elapsed_time_s=elapsed_time_s)
    write_json(
        run.checkpoint_file,
        {
            "cycle": int(cycle),
            "status": status,
            "flank_wear_mm": flank_wear_mm,
            "max_surface_move_um": max_surface_move_um,
            "elapsed_time_s": elapsed_time_s,
            "stop_reason": stop_reason,
        },
    )


def load_checkpoint(run: RunDirectory) -> Checkpoint:
    if not run.checkpoint_file.exists() or not run.state_file.exists():
        raise FileNotFoundError(f"no checkpoint in {run.path}")
    meta = read_json(run.checkpoint_file)
    with np.load(run.state_file) as data:
        coords = np.array(data["coords"])
        if int(data["cycle"]) != int(meta["cycle"]):
            raise ValueError("checkpoint.json and state.npz disagree on the cycle number")
    return Checkpoint(
        cycle=int(meta["cycle"]),
        status=str(meta["status"]),
        elapsed_time_s=float(meta["elapsed_time_s"]),
        flank_wear_mm=float(meta["flank_wear_mm"]),
        max_surface_move_um=float(meta["max_surface_move_um"]),
        stop_reason=meta.get("stop_reason"),
        coords=coords,
    )
