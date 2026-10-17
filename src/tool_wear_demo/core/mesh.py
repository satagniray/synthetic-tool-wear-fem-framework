"""Synthetic 2D orthogonal-cutting tool section and its structured quad mesh.

Coordinate convention (all lengths in mm)
-----------------------------------------
* The sharp (theoretical) tool tip sits at the origin.
* The flank face runs to the right (+x), inclined upwards by the clearance angle.
* The rake face runs upwards (+y), leaning towards +x by the rake angle.
* The cutting edge is rounded with a circular hone of ``edge_radius_mm``.
* The tool body is closed by a vertical back face (x = width), a horizontal
  top face (y = height) and a chamfer joining them; ``back_split`` sets where
  the chamfer starts on each face. These faces are clamped (non-wearing).
* The workpiece moves along the x axis underneath the tool, so the cutting
  velocity is parallel to x and its normal is +y.

Mesh construction
-----------------
The wear contour C(u) (flank -> rounded edge -> rake) and the back chamfer
D(u) are parameterised by the same normalised arc length. Nodes are
placed on straight lines between C(u) and D(u) (a ruled transfinite map),
which gives a structured quadrilateral mesh that is refined towards the
cutting edge and towards the wear surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np

WEAR_SURFACE_TYPES: tuple[str, ...] = ("flank", "edge", "rake")
SURFACE_TYPES: tuple[str, ...] = WEAR_SURFACE_TYPES + ("back", "interior")


def _from_mapping(cls, data: Mapping[str, Any]):
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**dict(data))


@dataclass(frozen=True)
class ToolGeometry:
    """Parameters of the synthetic tool cross-section."""

    rake_angle_deg: float = 8.0
    clearance_angle_deg: float = 6.0
    edge_radius_mm: float = 0.025
    width_mm: float = 0.8
    height_mm: float = 1.2
    back_split: float = 0.5

    def __post_init__(self) -> None:
        if not -20.0 <= self.rake_angle_deg <= 30.0:
            raise ValueError("rake_angle_deg must lie in [-20, 30]")
        if not 1.0 <= self.clearance_angle_deg <= 20.0:
            raise ValueError("clearance_angle_deg must lie in [1, 20]")
        if self.wedge_angle_deg < 40.0:
            raise ValueError("wedge angle (90 - rake - clearance) must be at least 40 deg")
        if not 0.0 < self.edge_radius_mm < 0.2 * min(self.width_mm, self.height_mm):
            raise ValueError("edge_radius_mm must be positive and small compared with the tool size")
        if not 0.1 <= self.back_split <= 0.9:
            raise ValueError("back_split must lie in [0.1, 0.9]")

    @property
    def wedge_angle_deg(self) -> float:
        return 90.0 - self.rake_angle_deg - self.clearance_angle_deg

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ToolGeometry":
        return _from_mapping(cls, data)


@dataclass(frozen=True)
class MeshSpec:
    """Resolution and grading of the structured mesh."""

    n_flank: int = 30
    n_edge: int = 8
    n_rake: int = 40
    n_depth: int = 30
    surface_growth: float = 1.08
    depth_exponent: float = 1.6

    def __post_init__(self) -> None:
        for name in ("n_flank", "n_rake", "n_depth"):
            if int(getattr(self, name)) < 3:
                raise ValueError(f"{name} must be >= 3")
        if int(self.n_edge) < 2:
            raise ValueError("n_edge must be >= 2")
        if not 1.0 <= self.surface_growth <= 1.5:
            raise ValueError("surface_growth must lie in [1.0, 1.5]")
        if not 1.0 <= self.depth_exponent <= 3.0:
            raise ValueError("depth_exponent must lie in [1.0, 3.0]")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MeshSpec":
        return _from_mapping(cls, data)

    @property
    def n_contour_elements(self) -> int:
        return int(self.n_flank) + int(self.n_edge) + int(self.n_rake)


@dataclass
class Mesh:
    """Minimal 2D quadrilateral mesh.

    Attributes
    ----------
    coords:
        ``(N, 2)`` nodal coordinates in mm. Row ``i`` is node ``i`` (public node id ``i + 1``).
    elements:
        ``(M, 4)`` zero-based node indices, counter-clockwise.
    surface_type:
        ``(N,)`` per-node tag: ``flank``, ``edge``, ``rake`` (wearing surface),
        ``back`` (clamped, non-wearing boundary) or ``interior``.
    grid_shape:
        ``(n_depth + 1, n_contour + 1)`` for structured meshes, else ``None``.
    """

    coords: np.ndarray
    elements: np.ndarray
    surface_type: np.ndarray
    grid_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 2)
        self.elements = np.asarray(self.elements, dtype=int).reshape(-1, 4)
        self.surface_type = np.asarray(self.surface_type, dtype="U8")
        if self.surface_type.shape != (self.coords.shape[0],):
            raise ValueError("surface_type must have one entry per node")
        if self.elements.size and (self.elements.min() < 0 or self.elements.max() >= self.n_nodes):
            raise ValueError("element connectivity references unknown nodes")
        unknown = set(np.unique(self.surface_type)) - set(SURFACE_TYPES)
        if unknown:
            raise ValueError(f"unknown surface types: {sorted(unknown)}")

    # -- basic properties -------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def node_ids(self) -> np.ndarray:
        """Public 1-based node ids (as used in CSV output)."""
        return np.arange(1, self.n_nodes + 1)

    @property
    def is_surface(self) -> np.ndarray:
        return self.surface_type != "interior"

    @property
    def wear_surface_mask(self) -> np.ndarray:
        return np.isin(self.surface_type, WEAR_SURFACE_TYPES)

    @property
    def fixed_mask(self) -> np.ndarray:
        """Nodes held fixed during the geometry update (clamped back faces)."""
        return self.surface_type == "back"

    # -- helpers -----------------------------------------------------------
    def with_coords(self, coords: np.ndarray) -> "Mesh":
        """Return a mesh with identical topology and new coordinates."""
        return Mesh(np.array(coords, dtype=float, copy=True), self.elements, self.surface_type, self.grid_shape)

    def copy(self) -> "Mesh":
        return self.with_coords(self.coords)

    def element_areas(self) -> np.ndarray:
        from .quality import signed_areas

        return signed_areas(self.coords, self.elements)

    def area(self) -> float:
        return float(self.element_areas().sum())

    def unique_edges(self) -> np.ndarray:
        """``(E, 2)`` array of unique undirected element edges."""
        edges = np.stack([self.elements, np.roll(self.elements, -1, axis=1)], axis=-1).reshape(-1, 2)
        return np.unique(np.sort(edges, axis=1), axis=0)


# ---------------------------------------------------------------------------
# geometry generation
# ---------------------------------------------------------------------------
def _graded(n: int, ratio: float) -> np.ndarray:
    """``n + 1`` parameters in [0, 1] whose spacing grows geometrically by ``ratio``."""
    k = np.arange(n + 1, dtype=float)
    if abs(ratio - 1.0) < 1e-12:
        return k / n
    return (ratio**k - 1.0) / (ratio**n - 1.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def wear_contour_curve(geometry: ToolGeometry, spec: MeshSpec) -> tuple[np.ndarray, list[str]]:
    """Return the ordered wear contour (flank end -> rounded edge -> rake end) and node tags.

    The two end points lie on the clamped back faces and are tagged ``back``.
    """
    a = math.radians(geometry.clearance_angle_deg)
    g = math.radians(geometry.rake_angle_deg)
    flank_dir = np.array([math.cos(a), math.sin(a)])
    rake_dir = np.array([math.sin(g), math.cos(g)])
    wedge = math.acos(float(np.clip(flank_dir @ rake_dir, -1.0, 1.0)))

    rho = geometry.edge_radius_mm
    tangent_offset = rho / math.tan(wedge / 2.0)
    t_flank = tangent_offset * flank_dir
    t_rake = tangent_offset * rake_dir
    centre = _unit(flank_dir + rake_dir) * rho / math.sin(wedge / 2.0)
    th_f = math.atan2(*(t_flank - centre)[::-1])
    th_r = math.atan2(*(t_rake - centre)[::-1])
    sweep = (th_r - th_f + math.pi) % (2.0 * math.pi) - math.pi  # short way round the tip
    angles = th_f + sweep * np.linspace(0.0, 1.0, int(spec.n_edge) + 1)
    arc = centre + rho * np.column_stack([np.cos(angles), np.sin(angles)])

    p_flank_end = flank_dir * (geometry.width_mm / math.cos(a))
    p_rake_end = rake_dir * (geometry.height_mm / math.cos(g))

    grade_f = _graded(int(spec.n_flank), spec.surface_growth)
    flank = t_flank + grade_f[::-1, None] * (p_flank_end - t_flank)  # flank end ... tangent point
    grade_r = _graded(int(spec.n_rake), spec.surface_growth)
    rake = t_rake + grade_r[:, None] * (p_rake_end - t_rake)  # tangent point ... rake end

    points = np.vstack([flank, arc[1:-1], rake])
    tags = (
        ["back"]
        + ["flank"] * (int(spec.n_flank) - 1)
        + ["edge"] * (int(spec.n_edge) + 1)
        + ["rake"] * (int(spec.n_rake) - 1)
        + ["back"]
    )
    assert len(tags) == points.shape[0]
    return points, tags


def create_tool_mesh(geometry: ToolGeometry | None = None, spec: MeshSpec | None = None) -> Mesh:
    """Create the synthetic tool mesh (structured quadrilaterals, counter-clockwise)."""
    geometry = geometry or ToolGeometry()
    spec = spec or MeshSpec()
    contour, tags = wear_contour_curve(geometry, spec)
    n_u = contour.shape[0] - 1
    n_v = int(spec.n_depth)

    corner = np.array([geometry.width_mm, geometry.height_mm])
    q0 = contour[0] + geometry.back_split * (corner - contour[0])
    q1 = contour[-1] + geometry.back_split * (corner - contour[-1])

    seg = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= s[-1]
    back = q0 + s[:, None] * (q1 - q0)  # straight chamfer between the two back faces

    v = (np.arange(n_v + 1) / n_v) ** spec.depth_exponent
    grid = (1.0 - v)[:, None, None] * contour[None] + v[:, None, None] * back[None]
    coords = grid.reshape(-1, 2)

    tag_grid = np.full((n_v + 1, n_u + 1), "interior", dtype="U8")
    tag_grid[0, :] = tags
    tag_grid[1:, 0] = "back"
    tag_grid[1:, -1] = "back"
    tag_grid[-1, :] = "back"

    j, i = np.meshgrid(np.arange(n_v), np.arange(n_u), indexing="ij")
    n00 = (j * (n_u + 1) + i).ravel()
    n10 = n00 + 1
    n01 = n00 + (n_u + 1)
    n11 = n01 + 1
    elements = np.column_stack([n00, n10, n11, n01])

    mesh = Mesh(coords, elements, tag_grid.ravel(), grid_shape=(n_v + 1, n_u + 1))
    if mesh.element_areas()[0] < 0:  # enforce counter-clockwise ordering
        mesh = Mesh(coords, np.column_stack([n00, n01, n11, n10]), tag_grid.ravel(), grid_shape=(n_v + 1, n_u + 1))
    return mesh


def mesh_from_config(config: Mapping[str, Any]) -> Mesh:
    """Build the initial mesh from the ``geometry`` and ``mesh`` config sections."""
    return create_tool_mesh(ToolGeometry.from_mapping(config["geometry"]), MeshSpec.from_mapping(config["mesh"]))
