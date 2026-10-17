"""Wear laws.

Usui-type wear-rate law (as used for FEM tool-wear estimation, e.g. Xie et al., 2005)::

    w_dot = C * sigma_n * v_s * exp(-lambda / theta)

``w_dot``   wear rate normal to the tool surface
``sigma_n`` normal contact pressure
``v_s``     relative sliding velocity
``theta``   absolute interface temperature
``C, lambda`` material-pair constants

Units used in this package: ``sigma_n`` [MPa], ``v_s`` [m/s], ``theta`` [K];
``C`` is chosen such that ``w_dot`` comes out in [mm/s].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class UsuiRegime:
    """Constants valid from ``min_temperature_K`` upwards."""

    min_temperature_K: float
    C: float
    lambda_K: float


@dataclass(frozen=True)
class UsuiParameters:
    """A set of (possibly temperature-dependent) Usui constants."""

    name: str
    regimes: tuple[UsuiRegime, ...]
    units: str
    note: str = ""
    verified_for_quantitative_use: bool = False

    def regime_constants(self, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pick ``C`` and ``lambda`` for every temperature (highest applicable regime)."""
        regimes = sorted(self.regimes, key=lambda r: r.min_temperature_K)
        C = np.full(temperature.shape, regimes[0].C, dtype=float)
        lam = np.full(temperature.shape, regimes[0].lambda_K, dtype=float)
        for regime in regimes[1:]:
            mask = temperature >= regime.min_temperature_K
            C[mask] = regime.C
            lam[mask] = regime.lambda_K
        return C, lam


@dataclass(frozen=True)
class SyntheticDemoParameters(UsuiParameters):
    """Illustrative, dimensionally consistent demo constants (NOT calibrated).

    ``C`` is picked so that the synthetic fields produce wear rates of a few
    micrometres per minute; ``lambda`` gives a moderate temperature sensitivity.
    """

    name: str = "synthetic_demo"
    regimes: tuple[UsuiRegime, ...] = (UsuiRegime(min_temperature_K=0.0, C=3.0e-7, lambda_K=2500.0),)
    units: str = "sigma [MPa], v_s [m/s], theta [K] -> w_dot [mm/s]"
    note: str = "Synthetic demonstration constants. Not derived from experiments; results are not physically validated."
    verified_for_quantitative_use: bool = False


@dataclass(frozen=True)
class UsuiPublishedParameters(UsuiParameters):
    """Two-regime constants for an uncoated carbide / plain-carbon-steel pair.

    These values are those commonly quoted in the FEM tool-wear literature that
    follows Usui et al. and is used by Xie et al. (2005). Unit conventions for
    ``C`` differ between secondary sources, so the values are provided for
    reference only and MUST be checked against the original publications before
    any quantitative use. The default demo never uses them.
    """

    name: str = "usui_literature_reference"
    regimes: tuple[UsuiRegime, ...] = (
        UsuiRegime(min_temperature_K=0.0, C=7.80e-9, lambda_K=5.302e3),
        UsuiRegime(min_temperature_K=1150.0, C=1.198e-2, lambda_K=2.195e4),
    )
    units: str = "as quoted in the literature - verify against the original source before use"
    note: str = (
        "Literature reference values (Xie et al., Wear 258 (2005) 1479-1490, and sources cited there). "
        "Not re-verified in this repository; check units before quantitative use."
    )
    verified_for_quantitative_use: bool = False


class WearModel(ABC):
    """Interface for local wear-rate laws."""

    name: str = "abstract"

    @abstractmethod
    def wear_rate(self, pressure, temperature, sliding_velocity) -> np.ndarray:
        """Return the local wear rate for each node."""


@dataclass
class UsuiWearModel(WearModel):
    """Usui-type wear law ``w_dot = C * sigma * v_s * exp(-lambda / theta)``."""

    parameters: UsuiParameters = field(default_factory=SyntheticDemoParameters)

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"usui[{self.parameters.name}]"

    def wear_rate(self, pressure, temperature, sliding_velocity) -> np.ndarray:
        sigma = np.asarray(pressure, dtype=float)
        theta = np.asarray(temperature, dtype=float)
        v = np.asarray(sliding_velocity, dtype=float)
        sigma, theta, v = np.broadcast_arrays(sigma, theta, v)
        if not np.all(np.isfinite(theta)) or np.any(theta <= 0.0):
            raise ValueError("temperature must be finite and strictly positive (absolute temperature in K)")
        if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(v))):
            raise ValueError("pressure and sliding velocity must be finite")
        sigma = np.clip(sigma, 0.0, None)  # tensile / no contact -> no wear
        v = np.abs(v)
        C, lam = self.parameters.regime_constants(theta)
        return C * sigma * v * np.exp(-lam / theta)


def build_wear_model(name: str) -> WearModel:
    """Factory used by the configuration (``wear.model``)."""
    if name == "usui_demo":
        return UsuiWearModel(SyntheticDemoParameters())
    if name == "usui_published":
        return UsuiWearModel(UsuiPublishedParameters())
    raise ValueError(f"unknown wear model '{name}'")


def time_step(rate_mm_s: np.ndarray, mode: str, target_max_displacement_mm: float, fixed_dt_s: float) -> float:
    """Cycle duration ``delta_t`` [s].

    ``auto_scaled``: choose ``delta_t`` so the largest nodal wear increment equals
    the target displacement. This is a NUMERICAL step-size control for a stable
    demonstration (like limiting the geometry update per FEM re-mesh), not a
    physical calibration.
    ``fixed``: use ``fixed_dt_s`` as given.
    """
    if mode == "fixed":
        return float(fixed_dt_s)
    if mode != "auto_scaled":
        raise ValueError(f"unknown time-step mode '{mode}'")
    peak = float(np.max(rate_mm_s)) if np.size(rate_mm_s) else 0.0
    return target_max_displacement_mm / peak if peak > 0.0 else 0.0


def smooth_along_contour(values: np.ndarray, passes: int) -> np.ndarray:
    """Binomial [1, 2, 1] / 4 filter applied ``passes`` times along the contour.

    Removes node-to-node oscillation in the wear increment (from noisy contact
    output) while preserving the overall profile. The end values are kept.
    """
    out = np.asarray(values, dtype=float).copy()
    for _ in range(int(passes)):
        if out.size < 3:
            break
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out
