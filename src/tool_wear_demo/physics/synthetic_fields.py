"""Synthetic, physically plausible process fields on the wear contour.

!!! These fields are SYNTHETIC. They are smooth analytical stand-ins for the
contact pressure, interface temperature and sliding velocity that a coupled
thermo-mechanical cutting simulation would deliver. They are not measured,
not calibrated, and not taken from any real simulation.

Qualitative features that are deliberately reproduced
-----------------------------------------------------
* Contact pressure peaks close to the cutting edge and decays along the
  tool-chip contact length on the rake face.
* Interface temperature peaks *behind* the edge on the rake face, which is
  what drives crater wear in the classic picture.
* Sliding velocity is zero at the stagnation point on the rounded edge, rises
  to the chip speed on the rake face (sticking -> sliding transition) and to the
  cutting speed on the flank.
* Flank contact is decided geometrically: contour nodes whose height above
  the current lowest contour point is smaller than an elastic-recovery gap
  rub on the machined surface (flank-land pressure, frictional heating and the
  full cutting speed as sliding velocity). As a flat flank wear land develops,
  more nodes fall inside the gap, so the contact land grows with wear - a
  simple but genuine geometry/field coupling across cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np

from ..core.contour import ToolContour
from ..core.state import ProcessFieldState


@dataclass(frozen=True)
class SyntheticFieldParameters:
    """Parameters of the synthetic field generator (all illustrative)."""

    ambient_temperature_K: float = 293.15
    peak_temperature_K: float = 1100.0
    max_contact_pressure_MPa: float = 1200.0
    cutting_speed_m_per_min: float = 120.0
    noise_fraction: float = 0.02
    chip_speed_ratio: float = 0.45
    rake_contact_length_mm: float = 0.30
    pressure_center_mm: float = 0.01
    pressure_width_rake_mm: float = 0.12
    pressure_width_flank_mm: float = 0.03
    temperature_center_mm: float = 0.10
    temperature_width_mm: float = 0.14
    sticking_length_mm: float = 0.04
    flank_sliding_transition_mm: float = 0.01
    flank_pressure_ratio: float = 0.55
    flank_temperature_ratio: float = 0.80
    flank_contact_gap_um: float = 2.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SyntheticFieldParameters":
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown synthetic field keys: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in data.items()})

    @property
    def cutting_speed_m_s(self) -> float:
        return self.cutting_speed_m_per_min / 60.0


def gaussian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-(((x - center) / width) ** 2))


def flank_contact_indicator(y_mm: np.ndarray, gap_um: float) -> np.ndarray:
    """Smooth contact indicator in [0, 1] with the machined surface.

    1 at the lowest contour point, falling linearly to 0 at ``gap_um`` above it.
    """
    gap = (y_mm - y_mm.min()) * 1000.0
    return np.clip(1.0 - gap / gap_um, 0.0, 1.0)


def pressure_profile(d: np.ndarray, contact: np.ndarray, p: SyntheticFieldParameters) -> np.ndarray:
    width = np.where(d >= p.pressure_center_mm, p.pressure_width_rake_mm, p.pressure_width_flank_mm)
    shape = gaussian(d, p.pressure_center_mm, width)  # asymmetric: wide on the rake, narrow on the flank
    shape = np.where(d > p.rake_contact_length_mm, 0.0, shape)  # chip leaves the tool
    shape = np.maximum(shape, p.flank_pressure_ratio * contact)
    shape[shape < 1e-4] = 0.0  # truncate Gaussian tails: no contact far from the edge
    return p.max_contact_pressure_MPa * shape


def temperature_profile(d: np.ndarray, contact: np.ndarray, p: SyntheticFieldParameters) -> np.ndarray:
    shape = gaussian(d, p.temperature_center_mm, p.temperature_width_mm)
    shape = np.maximum(shape, p.flank_temperature_ratio * contact)
    return p.ambient_temperature_K + (p.peak_temperature_K - p.ambient_temperature_K) * shape


def sliding_velocity_profile(d: np.ndarray, contact: np.ndarray, p: SyntheticFieldParameters) -> np.ndarray:
    v_chip = p.chip_speed_ratio * p.cutting_speed_m_s
    rake = v_chip * (1.0 - np.exp(-np.clip(d, 0.0, None) / p.sticking_length_mm))
    flank = p.cutting_speed_m_s * (1.0 - np.exp(np.clip(d, None, 0.0) / p.flank_sliding_transition_mm))
    profile = np.where(d >= 0.0, rake, flank)
    # nodes rubbing on the machined surface slide at the cutting speed
    return profile + contact * (p.cutting_speed_m_s - profile)


def generate_process_fields(
    contour: ToolContour,
    coords: np.ndarray,
    params: SyntheticFieldParameters,
    rng: np.random.Generator | None = None,
) -> ProcessFieldState:
    """Evaluate the synthetic fields on the current wear contour."""
    pts = contour.points(coords)
    d = contour.signed_edge_distance(coords)
    contact = flank_contact_indicator(pts[:, 1], params.flank_contact_gap_um)

    pressure = pressure_profile(d, contact, params)
    temperature = temperature_profile(d, contact, params)
    sliding = sliding_velocity_profile(d, contact, params)

    if rng is not None and params.noise_fraction > 0:
        n = d.size
        pressure = pressure * (1.0 + rng.normal(0.0, params.noise_fraction, size=n))
        rise = temperature - params.ambient_temperature_K
        temperature = params.ambient_temperature_K + rise * (1.0 + rng.normal(0.0, 0.5 * params.noise_fraction, size=n))
    pressure = np.clip(pressure, 0.0, None)
    temperature = np.clip(temperature, params.ambient_temperature_K, None)

    return ProcessFieldState(
        node_index=contour.node_index.copy(),
        surface_type=contour.surface_type.copy(),
        s_norm=contour.normalized_coordinate(coords),
        d_mm=d,
        x_mm=pts[:, 0].copy(),
        y_mm=pts[:, 1].copy(),
        pressure_MPa=pressure,
        temperature_K=temperature,
        sliding_velocity_m_s=sliding,
        source="synthetic",
    )
