import math

import numpy as np
import pytest

from deribit.pricing import Black76Model
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.observations import SurfaceObservation
from deribit.surface.results import ArbitrageReport, CalibratedSmile
from deribit.surface.svi import SVIParameters
from deribit.surface.term_structure import (
    build_time_derivative_segments,
    common_k_grid,
    dw_dtau_between,
    fit_snapshot_term_structure,
    measure_backbone,
    spot_backbone,
    time_derivative,
)


# ======================================================================
# Builders
# ======================================================================


def _smile(
    parameters: SVIParameters,
    *,
    tau: float,
    forward: float = 80_000.0,
    expiration_timestamp: int = 1_800_000_000_000,
    k_min: float = -0.40,
    k_max: float = 0.40,
) -> CalibratedSmile:
    """A CalibratedSmile built directly, bypassing the optimiser.

    The analytic identities under test hold exactly, so injecting calibration
    noise would only force loose tolerances and hide real errors.
    """
    return CalibratedSmile(
        snapshot_id=1,
        underlying_index="BTC-TEST",
        expiration_timestamp=expiration_timestamp,
        tau=tau,
        forward=forward,
        rate=0.0,
        parameters=parameters,
        observation_count=20,
        source_row_ids=tuple(range(20)),
        observed_k_min=k_min,
        observed_k_max=k_max,
        objective_value=0.0,
        rmse_total_variance=0.0,
        converged=True,
        optimizer_message="synthetic",
        starting_point_count=1,
        residuals=(),
        arbitrage_report=ArbitrageReport(
            butterfly_free=True,
            positive_variance=True,
            valid_wing_slopes=True,
            call_monotone=True,
            call_convex=True,
            prices_within_bounds=True,
            minimum_total_variance=float(parameters.minimum_total_variance),
            minimum_density_condition=1.0,
            minimum_convexity_margin=1.0,
            violations=(),
        ),
    )


def _synthetic_expiry_observations(
    *,
    expiration_timestamp: int,
    parameters: SVIParameters,
    tau: float,
    forward: float = 65_000.0,
    rate: float = 0.03,
    n_points: int = 15,
) -> list[SurfaceObservation]:
    index = forward * math.exp(-rate * tau)
    model = Black76Model()
    observations = []
    for source_id, k in enumerate(np.linspace(-0.45, 0.45, n_points), 1):
        strike = forward * math.exp(float(k))
        option_type = "put" if k < 0.0 else "call"
        total_variance = float(parameters.total_variance(k))
        iv = math.sqrt(total_variance / tau)
        mid_usd = model.price(forward, strike, tau, iv, rate, option_type)
        mid_coin = mid_usd / index
        observations.append(
            SurfaceObservation(
                snapshot_id=7,
                source_row_id=source_id,
                instrument_name=f"BTC-TEST-{expiration_timestamp}-{strike:.2f}",
                underlying_index="BTC-TEST",
                expiration_timestamp=expiration_timestamp,
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


def _scaled(parameters: SVIParameters, factor: float, *, shift: float = 0.0):
    """Scale total variance by `factor` and translate the smile by `shift` in k."""
    return SVIParameters(
        a=parameters.a * factor,
        b=parameters.b * factor,
        rho=parameters.rho,
        m=parameters.m - shift,
        eta=parameters.eta,
    )


_BASE = SVIParameters(a=0.02, b=0.10, rho=-0.30, m=0.01, eta=0.20)


# ======================================================================
# SVIParameters additions
# ======================================================================


def test_minimum_total_variance_k_is_where_the_slope_vanishes():
    for parameters in (
        _BASE,
        SVIParameters(0.03, 0.15, -0.60, 0.05, 0.25),
        SVIParameters(0.03, 0.15, 0.40, -0.02, 0.10),
    ):
        k_star = parameters.minimum_total_variance_k
        assert parameters.first_derivative(k_star) == pytest.approx(0.0, abs=1e-12)
        # And it really is the minimum value, matching the closed form.
        assert parameters.total_variance(k_star) == pytest.approx(
            parameters.minimum_total_variance
        )


def test_minimum_total_variance_k_differs_materially_from_m():
    # The reading "decreasing m shifts the minimum left" treats m as the
    # minimum. At a realistic BTC rho it is off by a wide margin.
    parameters = SVIParameters(0.02, 0.10, -0.30, 0.031, 0.20)
    assert parameters.minimum_total_variance_k == pytest.approx(0.0939, abs=1e-3)
    assert abs(parameters.minimum_total_variance_k - parameters.m) > 0.06


def test_atm_shape_diagnostics():
    assert _BASE.atm_total_variance() == pytest.approx(
        float(_BASE.total_variance(0.0))
    )
    assert _BASE.atm_curvature == pytest.approx(
        float(_BASE.second_derivative(0.0))
    )
    # Peak curvature b/eta occurs at k = m and bounds the ATM value.
    assert _BASE.atm_curvature <= _BASE.b / _BASE.eta + 1e-12
    assert float(_BASE.second_derivative(_BASE.m)) == pytest.approx(
        _BASE.b / _BASE.eta
    )


def test_vega_matches_black76_finite_difference():
    forward, tau, rate = 80_000.0, 0.35, 0.0
    model = Black76Model()
    h = 1e-6
    for k in (-0.30, -0.05, 0.0, 0.20, 0.45):
        strike = forward * math.exp(k)
        sigma = float(_BASE.implied_vol(k, tau))
        up = model.price(forward, strike, tau, sigma + h, rate, "call")
        down = model.price(forward, strike, tau, sigma - h, rate, "call")
        finite_difference = (up - down) / (2.0 * h)
        assert float(_BASE.vega(k, forward, tau)) == pytest.approx(
            finite_difference, rel=1e-6
        )


def test_vega_discount_factor_scales_linearly():
    discount = math.exp(-0.05 * 0.35)
    plain = float(_BASE.vega(0.1, 80_000.0, 0.35))
    discounted = float(
        _BASE.vega(0.1, 80_000.0, 0.35, discount_factor=discount)
    )
    assert discounted == pytest.approx(plain * discount)


def test_dvol_dlnspot_is_zero_under_sticky_strike():
    # R = 1 is the definition of sticky strike: fixed-strike vols do not move.
    response = _BASE.dvol_dlnspot_fixed_strike(
        np.linspace(-0.3, 0.3, 11), 0.25, skew_stickiness_ratio=1.0
    )
    assert np.all(response == 0.0)


def test_dvol_dlnspot_is_minus_skew_under_sticky_delta():
    k = np.linspace(-0.3, 0.3, 11)
    sticky_delta = _BASE.dvol_dlnspot_fixed_strike(
        k, 0.25, skew_stickiness_ratio=0.0
    )
    assert sticky_delta == pytest.approx(-_BASE.dvol_dk(k, 0.25))


# ======================================================================
# Time dynamics
# ======================================================================


def test_time_derivative_matches_finite_difference_of_the_interpolation():
    near = _smile(_BASE, tau=0.25)
    far = _smile(_scaled(_BASE, 2.4), tau=0.75, expiration_timestamp=1_900_000_000_000)

    segment = time_derivative(near, far, n_points=21)
    arrays = segment.as_arrays()
    k = arrays["log_moneyness"]

    # The interpolation the analytic derivative is taken of: w linear in tau.
    w_near = near.parameters.total_variance(k)
    dw_dtau = arrays["dw_dtau"]

    def sigma(tau):
        return np.sqrt((w_near + (tau - near.tau) * dw_dtau) / tau)

    h = 1e-7
    finite_difference = (sigma(near.tau + h) - sigma(near.tau - h)) / (2.0 * h)
    assert arrays["dvol_dtau"] == pytest.approx(finite_difference, rel=1e-6)

    # The split must add back up to the whole.
    assert (
        arrays["accumulation_term"] + arrays["annualisation_term"]
    ) == pytest.approx(arrays["dvol_dtau"])


def test_time_derivative_is_zero_when_the_vol_term_structure_is_flat():
    # w scaled by exactly the tau ratio => identical sigma(k) at both
    # maturities => no vol drift, despite total variance tripling.
    near = _smile(_BASE, tau=0.25)
    far = _smile(
        _scaled(_BASE, 3.0), tau=0.75, expiration_timestamp=1_900_000_000_000
    )

    arrays = time_derivative(near, far, n_points=41).as_arrays()

    assert arrays["dvol_dtau"] == pytest.approx(0.0, abs=1e-12)
    assert arrays["vol_points_per_calendar_day"] == pytest.approx(0.0, abs=1e-12)
    # Accumulation and annualisation are individually large and cancel.
    assert np.all(np.abs(arrays["accumulation_term"]) > 0.05)
    assert np.all(arrays["calendar_margin_relative"] > 0.0)


def test_calendar_margin_is_flagged_when_total_variance_inverts():
    near = _smile(_BASE, tau=0.25)
    # Far expiry with LOWER total variance: calendar arbitrage.
    far = _smile(
        _scaled(_BASE, 0.5), tau=0.75, expiration_timestamp=1_900_000_000_000
    )

    segment = time_derivative(near, far, n_points=21)

    assert not segment.calendar_arbitrage_free
    assert segment.minimum_calendar_margin_relative < 0.0


def test_vol_points_per_day_has_the_holding_sign_convention():
    # Normal term structure: vol falls as maturity shortens is NOT implied;
    # what is fixed is that the reported per-day number is the negation of
    # d(sigma)/d(tau), i.e. the drift from holding one more day.
    near = _smile(_BASE, tau=0.25)
    far = _smile(
        _scaled(_BASE, 2.4), tau=0.75, expiration_timestamp=1_900_000_000_000
    )
    arrays = time_derivative(near, far, n_points=11).as_arrays()
    assert arrays["vol_points_per_calendar_day"] == pytest.approx(
        -arrays["dvol_dtau"] / 365.0 * 100.0
    )


def test_time_derivative_requires_far_tau_strictly_greater():
    near = _smile(_BASE, tau=0.25)
    far = _smile(_scaled(_BASE, 2.0), tau=0.75)
    with pytest.raises(ValueError):
        time_derivative(far, near)


def test_build_segments_skips_non_overlapping_pairs_instead_of_raising():
    first = _smile(_BASE, tau=0.25, k_min=-0.40, k_max=-0.20)
    second = _smile(
        _scaled(_BASE, 2.0), tau=0.50, k_min=0.20, k_max=0.40,
        expiration_timestamp=1_900_000_000_000,
    )
    third = _smile(
        _scaled(_BASE, 3.0), tau=0.75, k_min=0.10, k_max=0.40,
        expiration_timestamp=1_950_000_000_000,
    )

    segments = build_time_derivative_segments([first, second, third])

    # first -> second has no shared k, second -> third does.
    assert len(segments) == 1
    assert segments[0].expiry_near == 1_900_000_000_000


def test_common_k_grid_raises_when_ranges_disjoint():
    near = _smile(_BASE, tau=0.25, k_min=-0.4, k_max=-0.2)
    far = _smile(_scaled(_BASE, 2.0), tau=0.75, k_min=0.2, k_max=0.4)
    with pytest.raises(ValueError):
        common_k_grid(near, far)


# ======================================================================
# Spot dynamics
# ======================================================================


def test_spot_backbone_vol_response_is_proportional_to_skew():
    # The structural point: under any translation-family rule the IV response
    # is a scalar multiple of the skew, so that panel carries no independent
    # information. The vega-weighted panel is the one that does.
    smile = _smile(_BASE, tau=0.25)
    arrays = spot_backbone(smile, skew_stickiness_ratio=0.0).as_arrays()

    ratio = arrays["vol_points_per_1pct_spot"] / arrays["dvol_dk"]
    assert ratio == pytest.approx(ratio[0])

    # Vega weighting is NOT proportional: it peaks near ATM and decays.
    pnl_ratio = (
        arrays["vol_pnl_per_1pct_spot_bp_of_forward"] / arrays["dvol_dk"]
    )
    assert not np.allclose(pnl_ratio, pnl_ratio[0], rtol=1e-3)


def test_spot_backbone_is_flat_at_sticky_strike():
    smile = _smile(_BASE, tau=0.25)
    arrays = spot_backbone(smile, skew_stickiness_ratio=1.0).as_arrays()
    assert arrays["vol_points_per_1pct_spot"] == pytest.approx(0.0)
    assert arrays["vol_pnl_per_1pct_spot_bp_of_forward"] == pytest.approx(0.0)


# ======================================================================
# Measured backbone
# ======================================================================


def _backbone_pair(*, shift: float, delta: float, tau_early: float, tau_late: float):
    """Two snapshots of one expiry, related by a known stickiness rule.

    The late smile is the early one scaled to the shorter tau (which keeps
    sigma(k) unchanged, so the vol term structure is flat and the time
    component vanishes) and translated in k by `shift`.

        shift = 0      smile frozen in k        -> sticky delta, R = 0
        shift = delta  smile frozen in strike   -> sticky strike, R = 1
    """
    forward_early = 80_000.0
    forward_late = forward_early * math.exp(delta)
    early = _smile(_BASE, tau=tau_early, forward=forward_early)
    late = _smile(
        _scaled(_BASE, tau_late / tau_early, shift=shift),
        tau=tau_late,
        forward=forward_late,
    )
    # A flat vol term structure means w grows linearly from the origin, so
    # dw/dtau = w / tau = sigma^2, which exactly cancels the annualisation.
    def dw_dtau(k):
        return early.parameters.total_variance(k) / tau_early

    return early, late, dw_dtau


def test_measure_backbone_recovers_sticky_strike():
    early, late, dw_dtau = _backbone_pair(
        shift=0.03, delta=0.03, tau_early=0.30, tau_late=0.28
    )

    result = measure_backbone(early, late, dw_dtau=dw_dtau, n_points=51)

    # Fixed-strike vols did not move at all: R = 1, exactly.
    assert result.vega_weighted_stickiness_ratio == pytest.approx(1.0, abs=1e-9)
    assert result.least_squares_stickiness_ratio == pytest.approx(1.0, abs=1e-9)
    observed = np.array([p.dvol_observed for p in result.points])
    assert observed == pytest.approx(0.0, abs=1e-12)


def test_measure_backbone_recovers_sticky_delta():
    delta = 0.002  # small, since R = 0 holds to first order in d(lnS)
    early, late, dw_dtau = _backbone_pair(
        shift=0.0, delta=delta, tau_early=0.30, tau_late=0.28
    )

    result = measure_backbone(early, late, dw_dtau=dw_dtau, n_points=51)

    assert result.vega_weighted_stickiness_ratio == pytest.approx(0.0, abs=1e-2)
    assert result.least_squares_stickiness_ratio == pytest.approx(0.0, abs=1e-2)


def test_measure_backbone_time_component_is_removed_not_ignored():
    early, late, dw_dtau = _backbone_pair(
        shift=0.03, delta=0.03, tau_early=0.30, tau_late=0.28
    )

    with_time = measure_backbone(early, late, dw_dtau=dw_dtau, n_points=51)
    without = measure_backbone(early, late, dw_dtau=None, n_points=51)

    # Dropping dw_dtau attributes pure annualisation drift to spot, which
    # moves R away from the truth. The note must say so.
    assert with_time.vega_weighted_stickiness_ratio == pytest.approx(1.0, abs=1e-9)
    assert abs(without.vega_weighted_stickiness_ratio - 1.0) > 1e-3
    assert "assumed 0" in without.identification_note


def test_measure_backbone_rejects_mismatched_expiry_and_direction():
    early, late, _ = _backbone_pair(
        shift=0.0, delta=0.02, tau_early=0.30, tau_late=0.28
    )
    with pytest.raises(ValueError):
        measure_backbone(late, early)  # late passed as early: tau increases

    other = _smile(_BASE, tau=0.28, expiration_timestamp=1_950_000_000_000)
    with pytest.raises(ValueError):
        measure_backbone(early, other)


def test_measure_backbone_rejects_an_unmoved_forward():
    early, late, _ = _backbone_pair(
        shift=0.0, delta=0.0, tau_early=0.30, tau_late=0.28
    )
    with pytest.raises(ValueError, match="not identified"):
        measure_backbone(early, late)


def test_dw_dtau_between_is_grid_independent():
    near = _smile(_BASE, tau=0.25)
    far = _smile(
        _scaled(_BASE, 2.4), tau=0.75, expiration_timestamp=1_900_000_000_000
    )
    callable_form = dw_dtau_between(near, far)

    # The same k must give the same value regardless of the grid it sits in,
    # which is the property a positional array does not have.
    coarse = np.linspace(-0.3, 0.3, 7)
    fine = np.linspace(-0.3, 0.3, 61)
    assert callable_form(coarse) == pytest.approx(callable_form(fine)[::10])

    expected = (
        far.parameters.total_variance(coarse)
        - near.parameters.total_variance(coarse)
    ) / (far.tau - near.tau)
    assert callable_form(coarse) == pytest.approx(expected)


def test_measure_backbone_rejects_a_callable_of_the_wrong_shape():
    early, late, _ = _backbone_pair(
        shift=0.0, delta=0.02, tau_early=0.30, tau_late=0.28
    )
    with pytest.raises(ValueError, match="expected"):
        measure_backbone(early, late, dw_dtau=lambda k: np.zeros(3), n_points=51)


# ======================================================================
# Fitting
# ======================================================================


def test_fit_snapshot_term_structure_sorts_by_tau_and_skips_thin_expiries():
    far_obs = _synthetic_expiry_observations(
        expiration_timestamp=1_900_000_000_000,
        parameters=SVIParameters(0.03, 0.12, -0.35, 0.0, 0.22),
        tau=0.75,
    )
    near_obs = _synthetic_expiry_observations(
        expiration_timestamp=1_800_000_000_000,
        parameters=SVIParameters(0.02, 0.10, -0.30, 0.0, 0.20),
        tau=0.25,
    )

    smiles = fit_snapshot_term_structure(
        {
            1_900_000_000_000: far_obs,
            1_800_000_000_000: near_obs,
            1_950_000_000_000: near_obs[:6],  # below min_observations
        },
        number_of_starts=16,
    )

    assert len(smiles) == 2
    assert smiles[0].tau < smiles[1].tau
    assert smiles[0].expiration_timestamp == 1_800_000_000_000


def test_fit_snapshot_term_structure_refuses_an_underdetermined_threshold():
    # 5 points and 5 free parameters gives RMSE ~ 0 and no fit-quality signal.
    with pytest.raises(ValueError, match="free SVI parameters"):
        fit_snapshot_term_structure({}, min_observations=5)


def test_fit_snapshot_term_structure_survives_a_failing_expiry():
    good = _synthetic_expiry_observations(
        expiration_timestamp=1_800_000_000_000,
        parameters=SVIParameters(0.02, 0.10, -0.30, 0.0, 0.20),
        tau=0.25,
    )
    # Same snapshot_id but inconsistent tau inside one expiry: calibration
    # raises, and that expiry must be dropped rather than killing the run.
    broken = [
        obs if index else type(obs)(**{**obs.__dict__, "tau": obs.tau * 2.0})
        for index, obs in enumerate(
            _synthetic_expiry_observations(
                expiration_timestamp=1_900_000_000_000,
                parameters=SVIParameters(0.02, 0.10, -0.30, 0.0, 0.20),
                tau=0.50,
            )
        )
    ]

    smiles = fit_snapshot_term_structure(
        {1_800_000_000_000: good, 1_900_000_000_000: broken},
        number_of_starts=8,
    )

    assert [s.expiration_timestamp for s in smiles] == [1_800_000_000_000]


def test_least_squares_ratio_is_stable_where_the_pointwise_average_is_not():
    """The pointwise estimator divides by the skew, which vanishes at the
    smile minimum. Place that minimum inside the fitted range and the
    pointwise average degrades while the regression stays put."""
    # rho = 0 puts the minimum at k = m = 0, i.e. right at the money, where
    # the vega weights are heaviest.
    flat_minimum = SVIParameters(a=0.02, b=0.10, rho=0.0, m=0.0, eta=0.20)
    tau_early, tau_late, delta = 0.30, 0.28, 0.03
    early = _smile(flat_minimum, tau=tau_early, forward=80_000.0)
    late = _smile(
        _scaled(flat_minimum, tau_late / tau_early, shift=delta),
        tau=tau_late,
        forward=80_000.0 * math.exp(delta),
    )

    def dw_dtau(k):
        return flat_minimum.total_variance(k) / tau_early

    result = measure_backbone(early, late, dw_dtau=dw_dtau, n_points=101)

    # Truth is sticky strike. Both should find it, and the regression must
    # not be the one that needs the looser tolerance.
    assert result.least_squares_stickiness_ratio == pytest.approx(1.0, abs=1e-9)
    # The pointwise curve genuinely has a pole here.
    ratios = np.array([p.implied_stickiness_ratio for p in result.points])
    assert np.isnan(ratios).any()


def test_least_squares_ratio_ignores_flat_skew_instead_of_diverging():
    # b -> small makes the whole smile nearly flat, so R is barely
    # identified anywhere. The estimator must stay finite rather than blow up.
    nearly_flat = SVIParameters(a=0.02, b=0.002, rho=-0.30, m=0.01, eta=0.20)
    tau_early, tau_late, delta = 0.30, 0.28, 0.03
    early = _smile(nearly_flat, tau=tau_early, forward=80_000.0)
    late = _smile(
        _scaled(nearly_flat, tau_late / tau_early, shift=delta),
        tau=tau_late,
        forward=80_000.0 * math.exp(delta),
    )

    def dw_dtau(k):
        return nearly_flat.total_variance(k) / tau_early

    result = measure_backbone(early, late, dw_dtau=dw_dtau, n_points=51)

    assert np.isfinite(result.least_squares_stickiness_ratio)
    assert result.least_squares_stickiness_ratio == pytest.approx(1.0, abs=1e-6)
