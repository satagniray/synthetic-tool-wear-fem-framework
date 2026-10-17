"""Process-field backends (synthetic default, optional solver stub)."""

from .abaqus_stub import AbaqusBackend
from .base import ProcessFieldProvider, SolverBackend
from .synthetic import SyntheticBackend

__all__ = ["AbaqusBackend", "ProcessFieldProvider", "SolverBackend", "SyntheticBackend"]
