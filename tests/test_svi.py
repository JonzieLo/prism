import math

import numpy as np
import pytest

from deribit.pricing import Black76Model
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.observations import SurfaceObservation
from deribit.surface.svi import SVIParameters


def _synthetic_observations() -> list[SurfaceObservation]:
    parameters = SVIParameters(
        a=0.025,
        b=0.12,
        rho=-0.35,
        m=0.02,
        eta=0.20,
    )
    forward = 65_000.0
    tau = 0.50
    rate = 0.03
    index = forward * math.exp(-rate * tau)
    model = Black76Model()
    observations = []
    for source_id, k in enumerate(np.linspace(-0.45, 0.45, 13), 1):
        strike = forward * math.exp(float(k))
        option_type = "put" if k < 0.0 else "call"
        total_variance = float(parameters.total_variance(k))
        iv = math.sqrt(total_variance / tau)
        mid_usd = model.price(
            forward,
            strike,
            tau,
            iv,
            rate,
            option_type,
        )
        mid_coin = mid_usd / index
        observations.append(
            SurfaceObservation(
                snapshot_id=7,
                source_row_id=source_id,
                instrument_name=f"BTC-TEST-{strike:.2f}",
                underlying_index="BTC-TEST",
                expiration_timestamp=1_800_000_000_000,
                index_price=index,
                forward=forward,
                strike=strike,
                tau=tau,
                rate=rate,
                log_moneyness=float(k),
                option_type=option_type,
                bid_coin=0.99 * mid_coin,
                ask_coin=1.01 * mid_coin,
                mid_coin=mid_coin,
                bid_usd=0.99 * mid_usd,
                ask_usd=1.01 * mid_usd,
                mid_usd=mid_usd,
                bid_iv=None,
                mid_iv=iv,
                ask_iv=None,
                total_variance=total_variance,
                relative_spread=0.02,
                open_interest=100.0,
                volume=10.0,
            )
        )
    return observations


def test_svi_formula_and_wing_slopes():
    parameters = SVIParameters(0.02, 0.10, -0.30, 0.01, 0.20)
    parameters.validate()
    k = np.array([-0.20, 0.0, 0.25])
    centered = k - parameters.m
    expected = parameters.a + parameters.b * (
        parameters.rho * centered
        + np.sqrt(centered**2 + parameters.eta**2)
    )
    assert parameters.total_variance(k) == pytest.approx(expected)
    assert parameters.left_wing_slope == pytest.approx(0.13)
    assert parameters.right_wing_slope == pytest.approx(0.07)


def test_svi_derivatives_match_finite_differences():
    parameters = SVIParameters(0.02, 0.10, -0.30, 0.01, 0.20)
    h_first = 1e-5
    h_second = 1e-4
    for k in (-0.30, 0.0, 0.40):
        first = (
            parameters.total_variance(k + h_first)
            - parameters.total_variance(k - h_first)
        ) / (2.0 * h_first)
        second = (
            parameters.total_variance(k + h_second)
            - 2.0 * parameters.total_variance(k)
            + parameters.total_variance(k - h_second)
        ) / (h_second * h_second)
        assert parameters.first_derivative(k) == pytest.approx(
            first,
            rel=1e-8,
            abs=1e-10,
        )
        assert parameters.second_derivative(k) == pytest.approx(
            second,
            rel=2e-7,
            abs=1e-7,
        )

def test_svi_volatility_dynamics_match_finite_differences():
    parameters = SVIParameters(0.02, 0.10, -0.30, 0.01, 0.20)
    tau = 0.50
    forward = 65_000.0
    spot = 64_000.0
    strike = 70_000.0
    k = math.log(strike / forward)
    h_k = 1e-5
    h_forward = 1.0
    h_spot = 1.0

    fd_k = (
        parameters.implied_vol(k + h_k, tau)
        - parameters.implied_vol(k - h_k, tau)
    ) / (2.0 * h_k)
    assert parameters.dvol_dk(k, tau) == pytest.approx(
        fd_k,
        rel=1e-8,
        abs=2e-10,
    )

    def vol_from_forward(value: float) -> float:
        return float(parameters.implied_vol(math.log(strike / value), tau))

    fd_forward = (
        vol_from_forward(forward + h_forward) - vol_from_forward(forward - h_forward)
        ) / (2.0 * h_forward)
    assert parameters.dvol_dforward_fixed_strike(k,forward,tau,) == pytest.approx(fd_forward, rel=1e-8, abs=1e-12)

    carry_ratio = forward / spot

    def vol_from_spot(value: float) -> float:
        moved_forward = carry_ratio * value
        return float(parameters.implied_vol(math.log(strike / moved_forward),tau))

    fd_spot = (
        vol_from_spot(spot + h_spot)- vol_from_spot(spot - h_spot)
        ) / (2.0 * h_spot)
    assert parameters.dvol_dspot_fixed_strike(k,spot,tau,) == pytest.approx(fd_spot, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize(
    "parameters",
    [
        SVIParameters(0.02, 0.0, 0.0, 0.0, 0.2),
        SVIParameters(0.02, 0.1, 1.0, 0.0, 0.2),
        SVIParameters(0.02, 0.1, 0.0, 0.0, 0.0),
        SVIParameters(-1.0, 0.1, 0.0, 0.0, 0.2),
    ],
)
def test_invalid_svi_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        parameters.validate()


def test_calibration_recovers_synthetic_curve_deterministically():
    observations = _synthetic_observations()
    first = calibrate_svi_slice(observations, number_of_starts=16)
    second = calibrate_svi_slice(observations, number_of_starts=16)

    k = np.array([item.log_moneyness for item in observations])
    market_w = np.array([item.total_variance for item in observations])
    fitted_w = first.parameters.total_variance(k)
    assert np.sqrt(np.mean((fitted_w - market_w) ** 2)) < 2e-5
    assert first.arbitrage_report.butterfly_free
    assert first.parameters == pytest.approx(second.parameters)
    assert first.objective_value == pytest.approx(second.objective_value)
    assert max(
        abs(item.spread_normalized_residual)
        for item in first.residuals
        if item.spread_normalized_residual is not None
    ) < 0.05
