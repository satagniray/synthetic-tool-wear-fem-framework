"""YAML configuration: defaults, loading, merging and validation.

Every default value below is an illustrative, synthetic demonstration value.
None of them is an experimental, calibrated or unpublished research value.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "project": {
        "name": "synthetic_tool_wear_demo",
        "seed": 42,
    },
    "geometry": {
        "rake_angle_deg": 8.0,
        "clearance_angle_deg": 6.0,
        "edge_radius_mm": 0.025,
        "width_mm": 0.8,
        "height_mm": 1.2,
        "back_split": 0.5,
    },
    "mesh": {
        "n_flank": 30,
        "n_edge": 8,
        "n_rake": 40,
        "n_depth": 30,
        "surface_growth": 1.08,
        "depth_exponent": 1.6,
    },
    "synthetic_fields": {
        "ambient_temperature_K": 293.15,
        "peak_temperature_K": 1100.0,
        "max_contact_pressure_MPa": 1200.0,
        "cutting_speed_m_per_min": 120.0,
        "noise_fraction": 0.02,
        "chip_speed_ratio": 0.45,
        "rake_contact_length_mm": 0.30,
        "pressure_center_mm": 0.01,
        "pressure_width_rake_mm": 0.12,
        "pressure_width_flank_mm": 0.03,
        "temperature_center_mm": 0.10,
        "temperature_width_mm": 0.14,
        "sticking_length_mm": 0.04,
        "flank_sliding_transition_mm": 0.01,
        "flank_pressure_ratio": 0.55,
        "flank_temperature_ratio": 0.80,
        "flank_contact_gap_um": 2.0,
    },
    "wear": {
        "model": "usui_demo",
        "flank_direction": "cutting_normal",
        "time_step_mode": "auto_scaled",
        "delta_t_s": 60.0,
        "target_max_surface_displacement_um": 4.0,
        "target_flank_wear_mm": 0.25,
        "max_cycles": 20,
        "vb_threshold_um": 0.5,
        "depth_smoothing_passes": 2,
    },
    "smoothing": {
        "method": "normalized_radial",
        "radius_mm": 0.08,
        "laplacian_iterations": 400,
        "redistribute_surface_nodes": True,
    },
    "quality": {
        "max_retries": 6,
        "retry_scale": 0.5,
        "min_scaled_jacobian": 0.05,
    },
    "output": {
        "directory": "results",
        "save_csv": True,
        "save_png": True,
        "save_html": True,
        "zoom_window_mm": 0.30,
    },
}

CHOICES: dict[tuple[str, str], tuple[str, ...]] = {
    ("wear", "model"): ("usui_demo", "usui_published"),
    ("wear", "flank_direction"): ("cutting_normal", "surface_normal"),
    ("wear", "time_step_mode"): ("auto_scaled", "fixed"),
    ("smoothing", "method"): ("normalized_radial", "laplacian"),
}

POSITIVE: tuple[tuple[str, str], ...] = (
    ("geometry", "edge_radius_mm"),
    ("geometry", "width_mm"),
    ("geometry", "height_mm"),
    ("synthetic_fields", "ambient_temperature_K"),
    ("synthetic_fields", "peak_temperature_K"),
    ("synthetic_fields", "max_contact_pressure_MPa"),
    ("synthetic_fields", "cutting_speed_m_per_min"),
    ("synthetic_fields", "pressure_width_rake_mm"),
    ("synthetic_fields", "pressure_width_flank_mm"),
    ("synthetic_fields", "temperature_width_mm"),
    ("synthetic_fields", "sticking_length_mm"),
    ("synthetic_fields", "flank_sliding_transition_mm"),
    ("synthetic_fields", "rake_contact_length_mm"),
    ("synthetic_fields", "flank_contact_gap_um"),
    ("wear", "delta_t_s"),
    ("wear", "target_max_surface_displacement_um"),
    ("wear", "target_flank_wear_mm"),
    ("wear", "max_cycles"),
    ("smoothing", "radius_mm"),
    ("smoothing", "laplacian_iterations"),
    ("output", "zoom_window_mm"),
)


class ConfigError(ValueError):
    """Raised for unknown keys or invalid values in a configuration."""


def default_config() -> dict[str, dict[str, Any]]:
    """Return a deep copy of the built-in default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``overrides`` into ``base`` section by section, rejecting unknown keys."""
    merged = copy.deepcopy(dict(base))
    for section, values in (overrides or {}).items():
        if section not in merged:
            raise ConfigError(f"unknown configuration section '{section}'")
        if not isinstance(values, Mapping):
            raise ConfigError(f"section '{section}' must be a mapping")
        for key, value in values.items():
            if key not in merged[section]:
                raise ConfigError(f"unknown key '{section}.{key}'")
            merged[section][key] = value
    return merged


def validate_config(config: Mapping[str, Any]) -> None:
    """Check enumerations and value ranges; raise :class:`ConfigError` on problems."""
    for (section, key), allowed in CHOICES.items():
        if config[section][key] not in allowed:
            raise ConfigError(f"'{section}.{key}' must be one of {allowed}, got {config[section][key]!r}")
    for section, key in POSITIVE:
        value = config[section][key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"'{section}.{key}' must be a positive number, got {value!r}")
    fields = config["synthetic_fields"]
    if fields["peak_temperature_K"] <= fields["ambient_temperature_K"]:
        raise ConfigError("peak_temperature_K must exceed ambient_temperature_K")
    if not 0 <= fields["noise_fraction"] < 0.5:
        raise ConfigError("noise_fraction must lie in [0, 0.5)")
    for key in ("chip_speed_ratio", "flank_pressure_ratio", "flank_temperature_ratio"):
        if not 0 <= fields[key] <= 1:
            raise ConfigError(f"synthetic_fields.{key} must lie in [0, 1]")
    quality = config["quality"]
    if not 0 < quality["retry_scale"] < 1:
        raise ConfigError("quality.retry_scale must lie in (0, 1)")
    if int(quality["max_retries"]) < 0:
        raise ConfigError("quality.max_retries must be >= 0")
    if not 0 <= quality["min_scaled_jacobian"] < 1:
        raise ConfigError("quality.min_scaled_jacobian must lie in [0, 1)")
    if int(config["wear"]["depth_smoothing_passes"]) < 0:
        raise ConfigError("wear.depth_smoothing_passes must be >= 0")
    if not isinstance(config["project"]["seed"], int):
        raise ConfigError("project.seed must be an integer")


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML file on top of the defaults, apply ``overrides`` and validate."""
    config = default_config()
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        config = merge_config(config, data)
    if overrides:
        config = merge_config(config, overrides)
    validate_config(config)
    return config


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a configuration to YAML (used to freeze the settings of a run)."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
