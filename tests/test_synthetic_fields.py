import numpy as np
import pytest

from tool_wear_demo.backends import AbaqusBackend, ProcessFieldProvider, SolverBackend, SyntheticBackend
from tool_wear_demo.physics.synthetic_fields import (
    SyntheticFieldParameters,
    flank_contact_indicator,
    generate_process_fields,
)


@pytest.fixture
def params():
    return SyntheticFieldParameters()


def test_fields_have_one_value_per_contour_node(small_mesh, contour, params):
    f = generate_process_fields(contour, small_mesh.coords, params)
    n = len(contour)
    for arr in (f.pressure_MPa, f.temperature_K, f.sliding_velocity_m_s, f.d_mm, f.s_norm):
        assert arr.shape == (n,)
    assert f.source == "synthetic"


def test_fields_are_physically_plausible(small_mesh, contour, params):
    f = generate_process_fields(contour, small_mesh.coords, params)
    assert np.all(f.pressure_MPa >= 0) and f.pressure_MPa.max() <= params.max_contact_pressure_MPa * 1.001
    assert np.all(f.temperature_K >= params.ambient_temperature_K)
    assert f.temperature_K.max() <= params.peak_temperature_K * 1.001
    assert np.all(f.sliding_velocity_m_s >= 0)
    assert f.sliding_velocity_m_s.max() <= params.cutting_speed_m_s + 1e-12
    assert f.s_norm[0] == 0.0 and f.s_norm[-1] == pytest.approx(1.0)


def test_temperature_peaks_on_rake_face(small_mesh, contour, params):
    f = generate_process_fields(contour, small_mesh.coords, params)
    assert f.d_mm[np.argmax(f.temperature_K)] > 0
    assert contour.surface_type[np.argmax(f.temperature_K)] in ("rake", "edge")


def test_no_pressure_beyond_chip_contact_length(small_mesh, contour, params):
    f = generate_process_fields(contour, small_mesh.coords, params)
    assert np.all(f.pressure_MPa[f.d_mm > params.rake_contact_length_mm] == 0)


def test_flank_slides_at_cutting_speed(small_mesh, contour, params):
    f = generate_process_fields(contour, small_mesh.coords, params)
    far_flank = f.d_mm < -0.1
    assert np.allclose(f.sliding_velocity_m_s[far_flank], params.cutting_speed_m_s, rtol=1e-3)


def test_contact_indicator():
    y = np.array([0.0, 0.001, 0.002, 0.004])
    c = flank_contact_indicator(y, gap_um=2.0)
    assert np.allclose(c, [1.0, 0.5, 0.0, 0.0])


def test_noise_is_seeded_and_deterministic(small_mesh, contour, params):
    b = SyntheticBackend(contour, params, seed=3)
    f1, f2 = b.run_cycle(small_mesh, 1), b.run_cycle(small_mesh, 1)
    assert np.array_equal(f1.pressure_MPa, f2.pressure_MPa)
    f3 = b.run_cycle(small_mesh, 2)
    assert not np.array_equal(f1.pressure_MPa, f3.pressure_MPa)


def test_backends_satisfy_protocols(contour, params):
    b = SyntheticBackend(contour, params)
    assert isinstance(b, SolverBackend) and isinstance(b, ProcessFieldProvider)


def test_abaqus_stub_is_excluded(small_mesh):
    with pytest.raises(NotImplementedError):
        AbaqusBackend().run_cycle(small_mesh, 1)


def test_unknown_parameter_rejected():
    with pytest.raises(ValueError):
        SyntheticFieldParameters.from_mapping({"not_a_parameter": 1.0})


@pytest.mark.abaqus
def test_abaqus_backend_runs_cycle():  # pragma: no cover - needs a licensed solver, never run in public CI
    pytest.skip("requires a local Abaqus installation and a solver adapter")
