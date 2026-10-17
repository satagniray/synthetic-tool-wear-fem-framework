"""Propagate wear-surface motion into the tool interior.

Moving only the surface nodes would crush the first element layer after a few
cycles. Both smoothers here extend the prescribed surface displacement into
the interior while

* keeping the clamped back faces fixed, and
* never moving an interior node further than the largest prescribed surface
  displacement (no artificial amplification where neighbourhoods overlap).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.mesh import Mesh


class InteriorSmoother(ABC):
    @abstractmethod
    def smooth(self, mesh: Mesh, source_index: np.ndarray, source_displacement: np.ndarray) -> np.ndarray:
        """Return the full ``(N, 2)`` displacement field.

        ``source_index`` / ``source_displacement`` prescribe the wear-surface
        nodes; nodes in ``mesh.fixed_mask`` get zero displacement.
        """


def _prepare(mesh: Mesh, source_index: np.ndarray, source_displacement: np.ndarray):
    source_index = np.asarray(source_index, dtype=int)
    source_displacement = np.asarray(source_displacement, dtype=float).reshape(-1, 2)
    if source_index.size != source_displacement.shape[0]:
        raise ValueError("one displacement vector per source node is required")
    disp = np.zeros_like(mesh.coords)
    disp[source_index] = source_displacement
    free = ~mesh.fixed_mask.copy()
    free[source_index] = False
    return source_index, source_displacement, disp, free


def smoothstep_decay(xi: np.ndarray) -> np.ndarray:
    """1 at ``xi = 0``, 0 for ``xi >= 1``, C1-continuous in between."""
    xi = np.clip(xi, 0.0, 1.0)
    return 1.0 - 3.0 * xi**2 + 2.0 * xi**3


class NormalizedRadialSmoother(InteriorSmoother):
    """Distance-weighted, *normalised* radial smoothing (default).

    For an interior node at distances ``d_i`` from the surface nodes::

        w_i  = (max(0, 1 - d_i / R) / d_i) ** 2        (tapered inverse distance)
        u    = decay(d_min / R) * sum(w_i * u_i) / sum(w_i)

    * The linear taper ``max(0, 1 - d/R)`` limits the influence to radius ``R``.
    * The inverse-distance factor makes a node directly beneath the surface
      follow its nearest surface nodes, so thin element layers are not crushed.
    * The normalised average is a convex combination of surface displacements,
      so overlapping neighbourhoods cannot add up (no artificial amplification).
    * The smooth decay factor fades the motion to zero at distance ``R``.

    Together this guarantees ``|u| <= max_i |u_i|``.
    """

    def __init__(self, radius_mm: float = 0.08, chunk_size: int = 4096) -> None:
        if radius_mm <= 0:
            raise ValueError("radius_mm must be positive")
        self.radius_mm = float(radius_mm)
        self.chunk_size = int(chunk_size)

    def smooth(self, mesh: Mesh, source_index: np.ndarray, source_displacement: np.ndarray) -> np.ndarray:
        source_index, source_disp, disp, free = _prepare(mesh, source_index, source_displacement)
        free_idx = np.flatnonzero(free)
        src = mesh.coords[source_index]
        R = self.radius_mm
        for start in range(0, free_idx.size, self.chunk_size):
            idx = free_idx[start : start + self.chunk_size]
            dist = np.linalg.norm(mesh.coords[idx, None, :] - src[None, :, :], axis=2)
            w = (np.clip(1.0 - dist / R, 0.0, None) / np.maximum(dist, 1e-12)) ** 2
            wsum = w.sum(axis=1)
            avg = np.divide(w @ source_disp, wsum[:, None], out=np.zeros((idx.size, 2)), where=wsum[:, None] > 0)
            disp[idx] = avg * smoothstep_decay(dist.min(axis=1) / R)[:, None]
        return disp


class LaplacianSmoother(InteriorSmoother):
    """Harmonic extension by Jacobi iteration (optional alternative).

    The surface displacement is prescribed, the back faces are held at zero and
    every free node repeatedly takes the average displacement of its edge
    neighbours. A discrete harmonic field obeys the maximum principle, so the
    interior motion is bounded by the prescribed boundary motion.
    """

    def __init__(self, iterations: int = 400, tolerance_mm: float = 1e-10) -> None:
        self.iterations = int(iterations)
        self.tolerance_mm = float(tolerance_mm)
        self._cache: tuple[int, np.ndarray, np.ndarray] | None = None

    def _neighbours(self, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
        key = id(mesh.elements)
        if self._cache is None or self._cache[0] != key:
            edges = mesh.unique_edges()
            lists: list[list[int]] = [[] for _ in range(mesh.n_nodes)]
            for a, b in edges.tolist():
                lists[a].append(b)
                lists[b].append(a)
            width = max(len(l) for l in lists)
            nbr = np.full((mesh.n_nodes, width), -1, dtype=int)
            for i, l in enumerate(lists):
                nbr[i, : len(l)] = l
            self._cache = (key, nbr, (nbr >= 0).sum(axis=1))
        return self._cache[1], self._cache[2]

    def smooth(self, mesh: Mesh, source_index: np.ndarray, source_displacement: np.ndarray) -> np.ndarray:
        _, _, disp, free = _prepare(mesh, source_index, source_displacement)
        nbr, count = self._neighbours(mesh)
        valid = nbr >= 0
        safe = np.where(valid, nbr, 0)
        free_idx = np.flatnonzero(free)
        for _ in range(self.iterations):
            gathered = disp[safe] * valid[..., None]
            new = gathered.sum(axis=1) / count[:, None]
            change = np.abs(new[free_idx] - disp[free_idx]).max() if free_idx.size else 0.0
            disp[free_idx] = new[free_idx]
            if change < self.tolerance_mm:
                break
        return disp


def build_smoother(config_smoothing: dict) -> InteriorSmoother:
    method = config_smoothing["method"]
    if method == "normalized_radial":
        return NormalizedRadialSmoother(radius_mm=config_smoothing["radius_mm"])
    if method == "laplacian":
        return LaplacianSmoother(iterations=config_smoothing["laplacian_iterations"])
    raise ValueError(f"unknown smoothing method '{method}'")
