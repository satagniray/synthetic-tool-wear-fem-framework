"""Small, dependency-free CSV helpers for column-oriented tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..core.mesh import Mesh


def _fmt(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    return str(value)


def write_table(path: str | Path, columns: Mapping[str, Sequence]) -> None:
    """Write equally long columns to ``path`` (header = column names)."""
    names = list(columns)
    arrays = [np.asarray(columns[n]) for n in names]
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"columns have different lengths: {lengths}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for row in zip(*arrays):
            writer.writerow([_fmt(v) for v in row])


def write_rows(path: str | Path, rows: Sequence[Mapping]) -> None:
    """Write a list of dicts (all with the same keys)."""
    if not rows:
        raise ValueError("no rows to write")
    write_table(path, {k: [r[k] for r in rows] for k in rows[0]})


def _convert(values: list[str]):
    for cast in (int, float):
        try:
            return np.array([cast(v) for v in values])
        except ValueError:
            continue
    return np.array(values)


def read_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read a CSV written by :func:`write_table` (numeric columns become arrays)."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    return {name: _convert([r[i] for r in rows]) for i, name in enumerate(header)}


def read_rows(path: str | Path) -> list[dict]:
    table = read_table(path)
    n = len(next(iter(table.values()))) if table else 0
    return [{k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in table.items()} for i in range(n)]


# -- mesh ----------------------------------------------------------------------
def write_nodes(path: str | Path, mesh: Mesh, coords: np.ndarray | None = None) -> None:
    xy = mesh.coords if coords is None else coords
    write_table(
        path,
        {
            "node_id": mesh.node_ids,
            "x_mm": xy[:, 0],
            "y_mm": xy[:, 1],
            "is_surface": mesh.is_surface.astype(int),
            "surface_type": mesh.surface_type,
        },
    )


def write_elements(path: str | Path, mesh: Mesh) -> None:
    el = mesh.elements + 1
    write_table(
        path,
        {"element_id": np.arange(1, mesh.n_elements + 1), "n1": el[:, 0], "n2": el[:, 1], "n3": el[:, 2], "n4": el[:, 3]},
    )


def read_mesh(nodes_path: str | Path, elements_path: str | Path) -> Mesh:
    nodes = read_table(nodes_path)
    elems = read_table(elements_path)
    order = np.argsort(nodes["node_id"])
    coords = np.column_stack([nodes["x_mm"][order], nodes["y_mm"][order]])
    connectivity = np.column_stack([elems["n1"], elems["n2"], elems["n3"], elems["n4"]]) - 1
    return Mesh(coords, connectivity, nodes["surface_type"][order])


def read_node_coords(path: str | Path) -> np.ndarray:
    nodes = read_table(path)
    order = np.argsort(nodes["node_id"])
    return np.column_stack([nodes["x_mm"][order], nodes["y_mm"][order]])
