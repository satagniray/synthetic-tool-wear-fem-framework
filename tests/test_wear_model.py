import numpy as np
import pytest

from tool_wear_demo.physics.wear_models import (
    SyntheticDemoParameters,
    UsuiPublishedParameters,
    UsuiWearModel,
    build_wear_model,
    smooth_along_contour,
    time_step,
)


@pytest.fixture
def model():
    return UsuiWearModel(SyntheticDemoParameters())


def test_positive_inputs_give_positive_wear(model):
    assert np.all(model.wear_rate([800.0, 1200.0], [900.0, 1100.0], [1.0, 2.0]) > 0)


def test_zero_pressure_gives_zero_wear(model):
    assert model.wear_rate(0.0, 1000.0, 2.0) == 0.0


def test_zero_sliding_velocity_gives_zero_wear(model):
    assert model.wear_rate(1000.0, 1000.0, 0.0) == 0.0


def test_negative_pressure_is_treated_as_no_contact(model):
    assert model.wear_rate(-50.0, 1000.0, 2.0) == 0.0


def test_wear_increases_with_temperature(model):
    temps = np.linspace(500.0, 1300.0, 9)
    rates = model.wear_rate(np.full(9, 1000.0), temps, np.full(9, 2.0))
    assert np.all(np.diff(rates) > 0)


def test_wear_is_linear_in_pressure_and_velocity(model):
    base = model.wear_rate(500.0, 1000.0, 1.0)
    assert model.wear_rate(1000.0, 1000.0, 1.0) == pytest.approx(2 * base)
    assert model.wear_rate(500.0, 1000.0, 3.0) == pytest.approx(3 * base)


def test_matches_usui_equation(model):
    regime = SyntheticDemoParameters().regimes[0]
    expected = regime.C * 900.0 * 1.5 * np.exp(-regime.lambda_K / 1050.0)
    assert model.wear_rate(900.0, 1050.0, 1.5) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0.0, -10.0, np.nan, np.inf])
def test_invalid_temperature_rejected(model, bad):
    with pytest.raises(ValueError):
        model.wear_rate(1000.0, bad, 1.0)


def test_published_parameters_switch_regime_at_threshold():
    params = UsuiPublishedParameters()
    C, lam = params.regime_constants(np.array([1000.0, 1150.0, 1300.0]))
    assert C[0] != C[1] and C[1] == C[2]
    assert lam[0] != lam[1]
    assert params.verified_for_quantitative_use is False


def test_factory():
    assert isinstance(build_wear_model("usui_demo"), UsuiWearModel)
    assert isinstance(build_wear_model("usui_published"), UsuiWearModel)
    with pytest.raises(ValueError):
        build_wear_model("archard")


def test_auto_scaled_time_step_hits_target():
    rate = np.array([1e-6, 4e-5, 2e-5])
    dt = time_step(rate, "auto_scaled", 0.004, 60.0)
    assert np.max(rate * dt) == pytest.approx(0.004)
    assert time_step(np.zeros(3), "auto_scaled", 0.004, 60.0) == 0.0
    assert time_step(rate, "fixed", 0.004, 60.0) == 60.0


def test_contour_filter_preserves_constant_and_ends():
    assert np.allclose(smooth_along_contour(np.full(7, 3.0), 3), 3.0)
    spiky = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    out = smooth_along_contour(spiky, 1)
    assert out[2] == pytest.approx(0.5) and out[0] == 0.0 and out.sum() == pytest.approx(1.0)
