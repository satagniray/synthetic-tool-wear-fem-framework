# Architecture

The framework is a small Python package (`src/tool_wear_demo`) organised so that
**physics**, **geometry**, **numerics** and the **source of the process fields**
can each be changed without touching the others.

## Data flow of one run

```text
Configuration (YAML)
    ↓
SyntheticBackend            (implements SolverBackend / ProcessFieldProvider)
    ↓
ProcessFieldState           (pressure, temperature, sliding velocity on the wear contour)
    ↓
WearModel                   (UsuiWearModel: wear rate per node)
    ↓
SurfaceUpdate               (wear depth × direction, tangential node relocation)
    ↓
InteriorSmoother            (NormalizedRadialSmoother | LaplacianSmoother)
    ↓
MeshQualityChecker          (accept, or halve the step and retry)
    ↓
CycleResult                 (updated mesh, metrics, per-cycle output)
    ↓
WearLoop                    (stopping criteria, history, checkpoint/resume, figures)
```

![workflow](images/fig1_workflow.png)

## Package layout and responsibilities

| Package | Module | Main objects | Responsibility |
|---|---|---|---|
| `core` | `mesh.py` | `ToolGeometry`, `MeshSpec`, `Mesh`, `create_tool_mesh` | Synthetic tool section, structured quad mesh, node tags (`flank`, `edge`, `rake`, `back`, `interior`) |
| | `contour.py` | `ToolContour`, `boundary_loop` | Extracts the outer boundary from element connectivity, orders the wear surface flank → edge → rake, computes tangents, inward normals, arc length, signed distance from the edge |
| | `quality.py` | `MeshQualityChecker`, `QualityReport` | Signed area, corner (scaled) Jacobians, angles, aspect ratio, edge length |
| | `state.py` | `ProcessFieldState`, `SimulationState` | Plain data containers passed between stages |
| `physics` | `synthetic_fields.py` | `SyntheticFieldParameters`, `generate_process_fields` | Analytical, explicitly synthetic process fields |
| | `wear_models.py` | `WearModel`, `UsuiWearModel`, `SyntheticDemoParameters`, `UsuiPublishedParameters`, `time_step` | Wear law, parameter sets, cycle-duration control, contour filter |
| | `directions.py` | `wear_directions` | Recession direction per node (rake normal, flank ⟂ cutting velocity, blended edge) |
| `displacement` | `surface_update.py` | `surface_displacement`, `redistribute_surface_nodes` | Wear-surface motion and tangential node relocation |
| | `interior_smoothing.py` | `NormalizedRadialSmoother`, `LaplacianSmoother` | Propagate surface motion into the body with bounded magnitude |
| `backends` | `base.py` | `SolverBackend`, `ProcessFieldProvider` (protocols) | The only contract the loop has with a "solver" |
| | `synthetic.py` | `SyntheticBackend` | Default, solver-free backend (seeded per cycle) |
| | `abaqus_stub.py` | `AbaqusBackend` | Documented placeholder - raises `NotImplementedError` |
| `workflow` | `cycle_runner.py` | `CycleRunner`, `CycleResult` | The 11-step single-cycle pipeline and its output files |
| | `wear_loop.py` | `WearLoop`, `build_cycle_runner` | Multi-cycle loop, stopping criteria, history, resume |
| | `checkpoints.py` | `save_checkpoint`, `load_checkpoint` | `checkpoint.json` + `state.npz` |
| | `metrics.py` | `compute_wear_metrics` | Recession, crater depth, flank-wear indicator, removed area |
| `io` | `config.py`, `csv_io.py`, `results.py` | `load_config`, `RunDirectory` | Validated YAML config, CSV tables, run-folder layout |
| `visualization` | `mesh_plot.py`, `wear_plot.py`, `dashboard.py` | `make_run_outputs` | Figures 1-6, GIF, Plotly dashboard |
| - | `cli.py` | `main` | `run`, `plot`, `validate`, `generate-demo-data` |

## Design principles

**Separation of physics.** The wear law only sees three arrays (pressure,
temperature, sliding velocity). A different law (e.g. Archard- or
Takeyama-Murata-type) is a new `WearModel` subclass; nothing else changes.

**Separation of geometry.** All geometric questions - which nodes form the wear
surface, in which order, where "inside" is - are answered by `ToolContour`
from the mesh connectivity. No node ids are hard-coded anywhere.

**Separation of the solver backend.** `CycleRunner` depends on the
`SolverBackend` protocol, not on a concrete solver. The synthetic backend is
the default; an adapter for a commercial FEM code would implement the same
`run_cycle(mesh, cycle) -> ProcessFieldState` method (see
`backends/abaqus_stub.py`) and reuse everything downstream.

**Reproducibility.** Every run freezes its configuration (`config_used.yaml`),
writes per-cycle CSV/JSON, and seeds the synthetic noise with `(seed, cycle)`.
A resumed run therefore reproduces an uninterrupted run exactly - this is
tested (`tests/test_multicycle.py::test_resume_reproduces_uninterrupted_run`).

**Testability.** Each stage is a pure function or a small class that can be
exercised on a coarse mesh in milliseconds; 74 tests run in about ten seconds
on every push (GitHub Actions, Python 3.11 and 3.12). Tests that would need a
commercial solver are marked `@pytest.mark.abaqus` and deselected by default.

## Run folder

```text
results/run_YYYYMMDD_HHMMSS/
├── config_used.yaml          frozen configuration
├── nodes_initial.csv         initial mesh (node_id, x, y, is_surface, surface_type)
├── elements.csv              connectivity (element_id, n1..n4)
├── history.csv               one row per cycle, cycle 0 = unworn
├── checkpoint.json           last completed cycle, status, stop reason
├── state.npz                 coordinates for resume
├── figures/                  fig1-fig6, wear_progression.gif, dashboard.html
└── cycle_001/
    ├── nodes_before.csv, nodes_after.csv
    ├── process_fields.csv, wear_rates.csv, displacement.csv
    ├── mesh_quality.json, summary.json
    └── geometry_before.png, geometry_after.png, process_fields.png
```
