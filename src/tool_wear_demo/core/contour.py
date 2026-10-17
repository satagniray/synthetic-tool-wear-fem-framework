"""Outer boundary extraction and the ordered wear contour (flank -> edge -> rake)."""

from __future__ import annotations

import numpy as np

from .mesh import WEAR_SURFACE_TYPES, Mesh


def boundary_loop(mesh: Mesh) -> np.ndarray:
    """Ordered outer-boundary node indices, counter-clockwise.

    Boundary edges are directed element edges whose reverse edge does not
    exist; with counter-clockwise elements they chain into a CCW loop.
    """
    el = mesh.elements
    directed = np.stack([el, np.roll(el, -1, axis=1)], axis=-1).reshape(-1, 2)
    edge_set = set(map(tuple, directed.tolist()))
    nxt: dict[int, int] = {}
    for a, b in directed.tolist():
        if (b, a) not in edge_set:
            if a in nxt:
                raise ValueError("non-manifold boundary: node has two outgoing boundary edges")
            nxt[a] = b
    if not nxt:
        raise ValueError("mesh has no boundary")
    start = min(nxt)
    loop = [start]
    current = nxt[start]
    while current != start:
        loop.append(current)
        current = nxt[current]
        if len(loop) > len(nxt):
            raise ValueError("boundary does not form a single closed loop")
    if len(loop) != len(nxt):
        raise ValueError("mesh boundary consists of more than one loop")
    loop_arr = np.array(loop)
    if polygon_area(mesh.coords[loop_arr]) < 0:
        loop_arr = loop_arr[::-1]
    return loop_arr


def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray-casting test (vectorised over ``points``)."""
    points = np.atleast_2d(points)
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = polygon[:, 0][None], polygon[:, 1][None]
    x1, y1 = np.roll(polygon[:, 0], -1)[None], np.roll(polygon[:, 1], -1)[None]
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (np.sum(crosses & (x < x_int), axis=1) % 2) == 1


class ToolContour:
    """The wearing part of the tool boundary as an ordered node chain.

    The topology (which nodes, in which order) is fixed when the contour is
    built; all geometric quantities are evaluated for the coordinates passed in,
    so the same object serves every wear cycle.
    """

    def __init__(self, mesh: Mesh) -> None:
        loop = boundary_loop(mesh)
        types = mesh.surface_type[loop]
        is_wear = np.isin(types, WEAR_SURFACE_TYPES)
        if not is_wear.any():
            raise ValueError("mesh has no wear-surface nodes on its boundary")
        n = loop.size
        starts = [k for k in range(n) if is_wear[k] and not is_wear[k - 1]]
        if len(starts) != 1:
            raise ValueError("the wear surface must be a single contiguous boundary segment")
        order = []
        k = starts[0]
        while is_wear[k % n] and len(order) < n:
            order.append(int(loop[k % n]))
            k += 1
        chain = np.array(order)
        anchors = (int(loop[(starts[0] - 1) % n]), int(loop[k % n]))  # fixed neighbours of the chain ends
        ccw = True
        if mesh.surface_type[chain[0]] == "rake" or mesh.surface_type[chain[-1]] == "flank":
            chain = chain[::-1]
            anchors = anchors[::-1]
            ccw = False
        self.node_index: np.ndarray = chain
        #: fixed (non-wearing) boundary nodes next to the flank end and the rake end of the chain
        self.anchor_index: tuple[int, int] = anchors
        self.surface_type: np.ndarray = mesh.surface_type[chain]
        self.boundary_index: np.ndarray = loop
        self._ccw = ccw
        if self.surface_type[0] not in ("flank", "edge") or self.surface_type[-1] not in ("rake", "edge"):
            raise ValueError("wear contour must run from the flank side to the rake side")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.node_index.size)

    def mask(self, surface_type: str) -> np.ndarray:
        return self.surface_type == surface_type

    def points(self, coords: np.ndarray) -> np.ndarray:
        return coords[self.node_index]

    def anchored_points(self, coords: np.ndarray) -> np.ndarray:
        """Wear contour including the two fixed anchor nodes at its ends."""
        a0, a1 = self.anchor_index
        return np.vstack([coords[a0], self.points(coords), coords[a1]])

    def anchored_fractions(self, coords: np.ndarray) -> np.ndarray:
        """Normalised arc-length position of every chain node on the anchored polyline."""
        p = self.anchored_points(coords)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        return (s / s[-1])[1:-1]

    def boundary_polygon(self, coords: np.ndarray) -> np.ndarray:
        return coords[self.boundary_index]

    def arclength(self, coords: np.ndarray) -> np.ndarray:
        p = self.points(coords)
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])

    def normalized_coordinate(self, coords: np.ndarray) -> np.ndarray:
        """Surface coordinate ``s`` in [0, 1] from the flank end to the rake end."""
        s = self.arclength(coords)
        return s / s[-1]

    def signed_edge_distance(self, coords: np.ndarray) -> np.ndarray:
        """Arc length measured from the middle of the rounded edge (flank < 0 < rake), mm."""
        s = self.arclength(coords)
        s_edge = s[self.mask("edge")]
        if s_edge.size == 0:
            raise ValueError("contour has no edge nodes")
        return s - 0.5 * (s_edge[0] + s_edge[-1])

    def edge_fraction(self, coords: np.ndarray) -> np.ndarray:
        """0 at the flank-side end of the rounded edge, 1 at the rake side, NaN elsewhere."""
        s = self.arclength(coords)
        out = np.full(s.shape, np.nan)
        m = self.mask("edge")
        se = s[m]
        span = se[-1] - se[0]
        out[m] = (se - se[0]) / span if span > 0 else 0.5
        return out

    def tangents(self, coords: np.ndarray) -> np.ndarray:
        """Unit tangents pointing from the flank side towards the rake side."""
        t = np.gradient(self.points(coords), axis=0)
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    def inward_normals(self, coords: np.ndarray) -> np.ndarray:
        """Unit normals pointing into the tool material."""
        t = self.tangents(coords)
        left = np.column_stack([-t[:, 1], t[:, 0]])
        return left if self._ccw else -left


def resample_polyline(polyline: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Points at normalised arc-length ``fractions`` along ``polyline``."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))])
    s /= s[-1]
    return np.column_stack([np.interp(fractions, s, polyline[:, 0]), np.interp(fractions, s, polyline[:, 1])])


def distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Shortest distance from each point to a polyline (vectorised)."""
    a = polyline[:-1][None]
    ab = (polyline[1:] - polyline[:-1])[None]
    ap = points[:, None, :] - a
    denom = np.broadcast_to(np.sum(ab * ab, axis=2), ap.shape[:2])
    num = np.sum(ap * ab, axis=2)
    t = np.clip(np.divide(num, denom, out=np.zeros_like(num), where=denom > 0), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)
