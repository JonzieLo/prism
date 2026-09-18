import pytest

from deribit.surface.arbitrage import (
    assert_butterfly_free,
    check_svi_slice_arbitrage,
    svi_density_condition,
)
from deribit.surface.svi import SVIParameters


def _check(parameters: SVIParameters):
    return check_svi_slice_arbitrage(
        parameters,
        forward=65_000.0,
        tau=0.50,
        rate=0.03,
        observed_k_min=-0.45,
        observed_k_max=0.45,
    )


def test_known_valid_svi_slice_is_butterfly_free():
    parameters = SVIParameters(0.025, 0.12, -0.35, 0.02, 0.20)
    report = _check(parameters)
    assert report.butterfly_free
    assert report.minimum_density_condition >= -1e-10
    assert report.minimum_convexity_margin >= -1e-10
    assert not report.violations
    assert_butterfly_free(report)


def test_excessive_wing_slope_is_reported():
    parameters = SVIParameters(0.10, 1.60, 0.50, 0.0, 0.30)
    report = _check(parameters)
    assert not report.butterfly_free
    assert not report.valid_wing_slopes
    assert any(
        item.check == "wing_slopes"
        for item in report.violations
    )
    with pytest.raises(AssertionError, match="butterfly arbitrage"):
        assert_butterfly_free(report)


def test_density_condition_returns_one_value_per_coordinate():
    parameters = SVIParameters(0.025, 0.12, -0.35, 0.02, 0.20)
    values = svi_density_condition(parameters, [-0.2, 0.0, 0.2])
    assert len(values) == 3
    assert all(value > 0.0 for value in values)
