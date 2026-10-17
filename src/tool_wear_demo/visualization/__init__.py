"""Figures, GIF animation and HTML dashboard."""

from .dashboard import load_run, make_run_outputs, write_dashboard
from .mesh_plot import plot_cycle_geometry, plot_displacement_heatmap, plot_initial_mesh
from .wear_plot import (
    make_wear_gif,
    plot_process_fields,
    plot_wear_indicator,
    plot_wear_progression,
    plot_workflow_diagram,
)

__all__ = [
    "load_run",
    "make_run_outputs",
    "write_dashboard",
    "plot_cycle_geometry",
    "plot_displacement_heatmap",
    "plot_initial_mesh",
    "make_wear_gif",
    "plot_process_fields",
    "plot_wear_indicator",
    "plot_wear_progression",
    "plot_workflow_diagram",
]
