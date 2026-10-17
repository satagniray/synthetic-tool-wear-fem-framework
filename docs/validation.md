# Validation status

> **This is a portfolio demonstrator using synthetic inputs and is not
> experimentally validated.** Its wear values, times and indicators must not be
> used to predict real tool life.

What *is* verified is the **software**: that the numerical workflow does what
it claims, robustly and reproducibly.

## Verified by automated tests (74 tests, `pytest -q`)

| Area | What is checked |
|---|---|
| Wear law | positive inputs → positive wear; zero pressure → zero; zero sliding → zero; tensile pressure → zero; monotonic in temperature; linear in pressure and velocity; equals the closed-form Usui expression; non-positive / non-finite temperature rejected; regime switch of the literature parameter set |
| Geometry & mesh | element count; all quads counter-clockwise and valid; surface-tag counts; edge nodes lie exactly on the hone radius; invalid geometry rejected; CSV round trip |
| Contour | closed CCW boundary loop built from connectivity; flank → edge → rake ordering; inward normals point inside the tool polygon (and outward probes lie outside); rake/flank normals match the geometry; signed edge distance; polyline helpers |
| Directions & smoothing | unit directions into the tool; displacement magnitude equals the requested depth; flank moves perpendicular to the cutting velocity; interior motion never exceeds the maximum surface motion (radial and Laplacian); overlapping neighbourhoods do not amplify; fixed nodes stay fixed; displacement decays with distance; node relocation restores spacing without changing the worn profile |
| Mesh quality | valid quad accepted; inverted, bow-tie and non-convex quads rejected; scaled-Jacobian threshold; deliberately oversized wear step is automatically reduced until valid; clean failure when retries are exhausted |
| Synthetic fields | shapes, bounds and plausibility; temperature peak on the rake; no pressure beyond the chip contact length; flank slides at cutting speed; contact indicator; seeded determinism; backends satisfy the protocols; Abaqus stub raises |
| Single cycle | wear happens, back faces stay fixed, tool area decreases, complete output folder, deterministic |
| Multi-cycle | 3-cycle integration (folders, monotonic wear, no inversion, checkpoint, final ≠ initial mesh); stops at the flank-wear target; **resumed run equals uninterrupted run**; in-memory mode; CLI `run` / `generate-demo-data` / `validate`; figures, GIF and dashboard; config validation |

The CLI also contains a self-check: `python -m tool_wear_demo.cli validate`.

## Plausibility of the default run (qualitative)

| Expectation from the cutting literature | Observed in the default run |
|---|---|
| Crater forms on the rake face *behind* the edge, where temperature peaks | yes - deepest point (41 µm) ≈ 90 µm along the rake face from the edge middle; half-depth width ≈ 45-135 µm |
| Flank wear forms a flat land parallel to the cutting direction | yes - by construction of the wear direction plus geometric contact |
| Flank-wear land width grows with time | yes - indicator 14 → 254 µm over 15 cycles |
| Wear indicators increase monotonically | yes - all recession metrics and the removed area |
| The mesh remains valid | yes - minimum scaled Jacobian decreases smoothly from 0.49 to 0.19, no retries needed |

## What would be needed for real validation

1. Replace the synthetic backend with a solver adapter that produces contact
   pressure, temperature and sliding velocity from a validated cutting model.
2. Use wear-law constants identified for the actual tool/work material pair.
3. Map the simulated cycle time to real cutting time.
4. Compare worn profiles and VB/KT values with measured tool-wear data
   (e.g. from optical or 3D-scanning measurements) at several cutting conditions.
5. Add re-meshing for long tool lives.
