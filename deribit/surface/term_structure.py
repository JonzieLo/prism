from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .calibration import calibrate_svi_slice
from .observations import SurfaceObservation
from .results import CalibratedSmile

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.0

# Relative, not absolute. w grows roughly linearly in tau, so an absolute
# tolerance is the wrong scale at long expiries, and -1e-14 is float noise
# rather than an arbitrage.
CALENDAR_ARBITRAGE_RELATIVE_TOLERANCE = -1e-8

# Raw SVI has 5 free parameters. Fitting 5 points gives an exactly determined
# system, RMSE ~ 0, and a completely uninformative fit-quality signal. Keep
# meaningful degrees of freedom.
DEFAULT_MINIMUM_OBSERVATIONS = 10


# ======================================================================
# Fitting
# ======================================================================


def fit_snapshot_term_structure(
    observations_by_expiry: dict[int, list[SurfaceObservation]],
    *,
    number_of_starts: int = 16,
    min_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
    require_butterfly_free: bool = True,
) -> list[CalibratedSmile]:
    """Calibrate one raw-SVI slice per expiry, sorted by tau.

    Unlike the previous version this logs every rejection and refuses to
    admit a static-arbitrageable slice into the term structure by default.
    Downstream calendar and dynamics calculations are meaningless on a
    slice that already fails butterfly.
    """
    if min_observations <= 5:
        raise ValueError(
            "min_observations must exceed the 5 free SVI parameters; "
            f"got {min_observations}"
        )

    smiles: list[CalibratedSmile] = []
    for expiry_ts, items in sorted(observations_by_expiry.items()):
        if len(items) < min_observations:
            logger.info(
                "expiry %s rejected: %d observations < %d required",
                expiry_ts,
                len(items),
                min_observations,
            )
            continue
        try:
            smile = calibrate_svi_slice(
                items, number_of_starts=number_of_starts
            )
        except Exception:
            logger.exception("expiry %s rejected: calibration failed", expiry_ts)
            continue
        if require_butterfly_free and not smile.arbitrage_report.butterfly_free:
            logger.warning(
                "expiry %s rejected: butterfly violated (min g = %.3e)",
                expiry_ts,
                smile.arbitrage_report.minimum_density_condition,
            )
            continue
        smiles.append(smile)
    return sorted(smiles, key=lambda smile: smile.tau)


def common_k_grid(
    near: CalibratedSmile,
    far: CalibratedSmile,
    *,
    n_points: int = 101,
) -> NDArray[np.float64]:
    k_min = max(near.observed_k_min, far.observed_k_min)
    k_max = min(near.observed_k_max, far.observed_k_max)
    if k_min >= k_max:
        raise ValueError(
            "Calibrated smiles do not share an observed log-moneyness range"
        )
    return np.linspace(k_min, k_max, n_points)


# ======================================================================
# Time dynamics:  d(sigma) / d(tau)
# ======================================================================


@dataclass(frozen=True)
class TimeDerivativePoint:
    log_moneyness: float
    implied_vol: float
    dw_dtau: float
    dvol_dtau: float
    accumulation_term: float
    annualisation_term: float
    vol_points_per_calendar_day: float
    calendar_margin_absolute: float
    calendar_margin_relative: float


@dataclass(frozen=True)
class TimeDerivativeSegment:
    """Time dynamics between two adjacent calibrated expiries.

    tau_near / tau_far live here rather than on every grid point.
    """

    expiry_near: int
    expiry_far: int
    tau_near: float
    tau_far: float
    tau_evaluated: float
    points: list[TimeDerivativePoint]

    @property
    def minimum_calendar_margin_relative(self) -> float:
        return min(point.calendar_margin_relative for point in self.points)

    @property
    def calendar_arbitrage_free(self) -> bool:
        return (
            self.minimum_calendar_margin_relative
            >= CALENDAR_ARBITRAGE_RELATIVE_TOLERANCE
        )

    def as_arrays(self) -> dict[str, NDArray[np.float64]]:
        return {
            name: np.array([getattr(p, name) for p in self.points])
            for name in (
                "log_moneyness",
                "implied_vol",
                "dw_dtau",
                "dvol_dtau",
                "accumulation_term",
                "annualisation_term",
                "vol_points_per_calendar_day",
                "calendar_margin_relative",
            )
        }


def time_derivative(
    near: CalibratedSmile,
    far: CalibratedSmile,
    *,
    n_points: int = 201,
    evaluate_at: str = "near",
) -> TimeDerivativeSegment:
    """Analytic d(sigma)/d(tau) from a calendar-arb-preserving interpolation.

    Raw SVI is fitted per slice and carries no tau dependence, so there is
    no native time derivative. Interpolating total variance LINEARLY in tau
    at fixed k is the interpolation that preserves calendar-arbitrage
    freeness, and it gives

        w(k, tau) = w_near(k) + (tau - tau_near) * dw_dtau
        dw_dtau   = (w_far(k) - w_near(k)) / (tau_far - tau_near)

    and, since sigma = sqrt(w / tau),

        d(sigma)/d(tau) =  dw_dtau / (2 sigma tau)  -  sigma / (2 tau)
                           ^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^
                           accumulation               annualisation

    The two terms oppose each other. As expiry approaches you give back
    accumulated variance (first term, normally positive, so vol falls) but
    you divide by a smaller tau (second term, negative, so vol rises).
    Which dominates is k- and tau-dependent, and collapsing them into one
    secant throws away exactly the thing worth looking at.

    This supersedes the previous `calendar_slope`, which returned an
    unattributed secant (sigma_far - sigma_near)/dtau evaluated at no
    particular tau.

    Note the convention: dvol_dtau is with respect to INCREASING maturity.
    `vol_points_per_calendar_day` is the negation, i.e. the vol drift you
    experience holding the position for one more day.
    """
    if far.tau <= near.tau:
        raise ValueError("`far` must have a strictly greater tau than `near`")
    if evaluate_at not in {"near", "far", "midpoint"}:
        raise ValueError(f"Unsupported evaluate_at: {evaluate_at}")

    k_grid = common_k_grid(near, far, n_points=n_points)
    w_near = near.parameters.total_variance(k_grid)
    w_far = far.parameters.total_variance(k_grid)
    d_tau = far.tau - near.tau
    dw_dtau = (w_far - w_near) / d_tau

    tau_eval = {
        "near": near.tau,
        "far": far.tau,
        "midpoint": 0.5 * (near.tau + far.tau),
    }[evaluate_at]
    w_eval = w_near + (tau_eval - near.tau) * dw_dtau
    sigma_eval = np.sqrt(np.maximum(w_eval, 1e-12) / tau_eval)

    accumulation = dw_dtau / (2.0 * sigma_eval * tau_eval)
    annualisation = -sigma_eval / (2.0 * tau_eval)
    dvol_dtau = accumulation + annualisation

    margin_absolute = w_far - w_near
    margin_relative = margin_absolute / np.maximum(w_near, 1e-12)

    points = [
        TimeDerivativePoint(
            log_moneyness=float(k_grid[i]),
            implied_vol=float(sigma_eval[i]),
            dw_dtau=float(dw_dtau[i]),
            dvol_dtau=float(dvol_dtau[i]),
            accumulation_term=float(accumulation[i]),
            annualisation_term=float(annualisation[i]),
            vol_points_per_calendar_day=float(
                -dvol_dtau[i] / DAYS_PER_YEAR * 100.0
            ),
            calendar_margin_absolute=float(margin_absolute[i]),
            calendar_margin_relative=float(margin_relative[i]),
        )
        for i in range(len(k_grid))
    ]

    return TimeDerivativeSegment(
        expiry_near=near.expiration_timestamp,
        expiry_far=far.expiration_timestamp,
        tau_near=near.tau,
        tau_far=far.tau,
        tau_evaluated=tau_eval,
        points=points,
    )


def build_time_derivative_segments(
    smiles: list[CalibratedSmile],
    *,
    n_points: int = 201,
    evaluate_at: str = "near",
) -> list[TimeDerivativeSegment]:
    """Adjacent-pair time dynamics, skipping pairs with no shared k range.

    The previous code let a ValueError from common_k_grid propagate out of
    the plotting function, destroying every panel for every pair.
    """
    segments: list[TimeDerivativeSegment] = []
    for near, far in zip(smiles, smiles[1:]):
        try:
            segments.append(
                time_derivative(
                    near, far, n_points=n_points, evaluate_at=evaluate_at
                )
            )
        except ValueError as error:
            logger.warning(
                "skipping pair %s -> %s: %s",
                near.expiration_timestamp,
                far.expiration_timestamp,
                error,
            )
    return segments


# ======================================================================
# Spot dynamics:  d(sigma) / d(ln S)
# ======================================================================


@dataclass(frozen=True)
class SpotBackbonePoint:
    log_moneyness: float
    implied_vol: float
    dvol_dk: float
    dvol_dlnspot: float
    vol_points_per_1pct_spot: float
    vega_per_unit_forward: float
    vol_pnl_per_1pct_spot_bp_of_forward: float


@dataclass(frozen=True)
class SpotBackbone:
    expiration_timestamp: int
    tau: float
    forward: float
    skew_stickiness_ratio: float
    forward_spot_elasticity: float
    points: list[SpotBackbonePoint]

    def as_arrays(self) -> dict[str, NDArray[np.float64]]:
        return {
            name: np.array([getattr(p, name) for p in self.points])
            for name in (
                "log_moneyness",
                "implied_vol",
                "dvol_dk",
                "dvol_dlnspot",
                "vol_points_per_1pct_spot",
                "vega_per_unit_forward",
                "vol_pnl_per_1pct_spot_bp_of_forward",
            )
        }


def spot_backbone(
    smile: CalibratedSmile,
    *,
    skew_stickiness_ratio: float = 0.0,
    forward_spot_elasticity: float = 1.0,
    n_points: int = 201,
    shock: float = 0.01,
) -> SpotBackbone:
    """Fixed-strike vol response to a spot move, under a stickiness rule.

    `vol_points_per_1pct_spot` is the IV move alone. It is a scalar multiple
    of d(sigma)/dk for ANY translation-family stickiness rule, so on its own
    it duplicates the skew panel. The quantity that carries independent
    information is `vol_pnl_per_1pct_spot_bp_of_forward` = vega * d(sigma),
    because vega peaks near ATM and dies in the wings.
    """
    k_grid = np.linspace(smile.observed_k_min, smile.observed_k_max, n_points)
    parameters = smile.parameters

    dvol_dk = parameters.dvol_dk(k_grid, smile.tau)
    dvol_dlnspot = parameters.dvol_dlnspot_fixed_strike(
        k_grid,
        smile.tau,
        skew_stickiness_ratio=skew_stickiness_ratio,
        forward_spot_elasticity=forward_spot_elasticity,
    )
    d_sigma = dvol_dlnspot * shock
    vega = parameters.vega(k_grid, smile.forward, smile.tau)
    vol_pnl_bp = (vega * d_sigma) / smile.forward * 10_000.0

    points = [
        SpotBackbonePoint(
            log_moneyness=float(k_grid[i]),
            implied_vol=float(parameters.implied_vol(k_grid, smile.tau)[i]),
            dvol_dk=float(dvol_dk[i]),
            dvol_dlnspot=float(dvol_dlnspot[i]),
            vol_points_per_1pct_spot=float(d_sigma[i] * 100.0),
            vega_per_unit_forward=float(vega[i] / smile.forward),
            vol_pnl_per_1pct_spot_bp_of_forward=float(vol_pnl_bp[i]),
        )
        for i in range(len(k_grid))
    ]

    return SpotBackbone(
        expiration_timestamp=smile.expiration_timestamp,
        tau=smile.tau,
        forward=smile.forward,
        skew_stickiness_ratio=skew_stickiness_ratio,
        forward_spot_elasticity=forward_spot_elasticity,
        points=points,
    )


# ======================================================================
# Empirical backbone: measure R instead of assuming it
# ======================================================================


def dw_dtau_between(
    near: CalibratedSmile,
    far: CalibratedSmile,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """d(w)/d(tau) at fixed k, as a callable, from two expiries of ONE snapshot.

    Returns a function rather than an array so the consumer evaluates it on
    its own k grid. `measure_backbone` works on the fixed-strike overlap of
    two SNAPSHOTS, while a TimeDerivativeSegment is gridded on the overlap of
    two EXPIRIES; passing the segment's array across that boundary lines up
    equal-length vectors sampled at different k.

    The linear-in-tau reading of w this implies is the same one
    `time_derivative` uses, and is the interpolation that preserves calendar
    arbitrage freeness.
    """
    if far.tau <= near.tau:
        raise ValueError("`far` must have a strictly greater tau than `near`")
    d_tau = far.tau - near.tau
    near_parameters = near.parameters
    far_parameters = far.parameters

    def _dw_dtau(k: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            far_parameters.total_variance(k)
            - near_parameters.total_variance(k)
        ) / d_tau

    return _dw_dtau



@dataclass(frozen=True)
class MeasuredBackbonePoint:
    log_moneyness_early: float
    log_moneyness_late: float
    dvol_observed: float
    dvol_time_component: float
    dvol_spot_component: float
    dvol_dk_early: float
    implied_stickiness_ratio: float
    vega_weight: float


@dataclass(frozen=True)
class MeasuredBackbone:
    expiration_timestamp: int
    d_ln_spot: float
    d_tau: float
    tau_early: float
    tau_late: float
    vega_weighted_stickiness_ratio: float
    least_squares_stickiness_ratio: float
    """Vega-weighted least-squares fit of the whole R(k) curve.

    Prefer this to the pointwise average. R(k) is recovered by dividing by
    d(sigma)/dk, which vanishes at the smile minimum -- and on a BTC fit that
    minimum sits near ATM, which is exactly where the vega weights are
    largest. The pointwise average therefore puts its heaviest weight on its
    worst-conditioned points. The regression form never divides by the skew.
    """
    identification_note: str
    points: list[MeasuredBackbonePoint]


def measure_backbone(
    early: CalibratedSmile,
    late: CalibratedSmile,
    *,
    dw_dtau: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    | NDArray[np.float64]
    | float
    | None = None,
    forward_spot_elasticity: float = 1.0,
    n_points: int = 201,
    minimum_abs_dvol_dk: float = 1e-3,
) -> MeasuredBackbone:
    """Back out the realised skew-stickiness ratio from two snapshots.

    `early` and `late` must be calibrations of the SAME expiry at two
    different snapshot times, so tau_late < tau_early.

    Decomposition, at a FIXED strike K:

        d(sigma_K) = (d sigma / d ln S) * d(ln S)
                   + (d sigma / d tau)  * d(tau)
                   + residual

    Two snapshots give one equation and two unknowns. This function
    identifies the spot term by SUBTRACTING a time term computed from
    `dw_dtau`, which should come from the term structure of the EARLY
    snapshot (build_time_derivative_segments on the early snapshot's
    smiles, then take dw_dtau for the segment starting at this expiry).

    If dw_dtau is None the fallback assumes total variance is frozen in k,
    i.e. dw_dtau = 0, so all of the time effect is annualisation. That is
    almost certainly wrong and the returned identification_note says so.

    The result is an identification, not a measurement. It becomes a real
    estimate only with many snapshots and a regression of fixed-strike vol
    changes on d(ln S) and d(tau).
    """
    if late.expiration_timestamp != early.expiration_timestamp:
        raise ValueError("Both smiles must be the same expiry")
    if late.tau >= early.tau:
        raise ValueError("`late` must be the later snapshot (smaller tau)")

    d_ln_spot = float(np.log(late.forward / early.forward))
    if abs(d_ln_spot) < 1e-6:
        raise ValueError(
            "Forwards are effectively unchanged between snapshots; the "
            "spot component is not identified"
        )
    d_tau = late.tau - early.tau

    k_early = np.linspace(
        max(early.observed_k_min, late.observed_k_min + d_ln_spot),
        min(early.observed_k_max, late.observed_k_max + d_ln_spot),
        n_points,
    )
    if k_early[0] >= k_early[-1]:
        raise ValueError("Snapshots share no common fixed-strike range")

    # Same strike, expressed in each snapshot's own forward coordinates.
    k_late = k_early - forward_spot_elasticity * d_ln_spot

    sigma_early = early.parameters.implied_vol(k_early, early.tau)
    sigma_late = late.parameters.implied_vol(k_late, late.tau)
    dvol_observed = sigma_late - sigma_early

    w_early = early.parameters.total_variance(k_early)
    if dw_dtau is None:
        dw_dtau_array = np.zeros_like(k_early)
        note = (
            "dw_dtau not supplied: assumed 0 (total variance frozen in k). "
            "The time component is annualisation only and the implied R is "
            "biased. Supply dw_dtau from the early snapshot's term structure."
        )
    elif callable(dw_dtau):
        dw_dtau_array = np.asarray(dw_dtau(k_early), dtype=float)
        if dw_dtau_array.shape != k_early.shape:
            raise ValueError(
                "dw_dtau callable returned shape "
                f"{dw_dtau_array.shape}, expected {k_early.shape}"
            )
        note = (
            "Time component removed using dw_dtau from the early snapshot's "
            "term structure. Two snapshots identify, they do not measure."
        )
    else:
        # An array is positional on THIS function's k grid, which is the
        # fixed-strike overlap of the two snapshots -- NOT the grid of a
        # TimeDerivativeSegment, which is the overlap of two expiries in one
        # snapshot. Those two grids have different endpoints, so handing over
        # `[p.dw_dtau for p in segment.points]` lines up 201 values against
        # 201 different k. Prefer `dw_dtau_between(near, far)`, which is
        # evaluated here and cannot be misaligned.
        dw_dtau_array = np.broadcast_to(
            np.asarray(dw_dtau, dtype=float), k_early.shape
        )
        note = (
            "Time component removed using a POSITIONAL dw_dtau array. This "
            "assumes the array was sampled on this function's own k grid; "
            "an array taken from a TimeDerivativeSegment is on a different "
            "grid and is silently misaligned. Prefer dw_dtau_between()."
        )

    accumulation = dw_dtau_array / (2.0 * sigma_early * early.tau)
    annualisation = -sigma_early / (2.0 * early.tau)
    dvol_dtau = accumulation + annualisation
    dvol_time = dvol_dtau * d_tau
    dvol_spot = dvol_observed - dvol_time

    dvol_dk_early = early.parameters.dvol_dk(k_early, early.tau)

    # d(sigma_K)/d(ln S) = eps (R - 1) d(sigma)/dk
    #   =>  R = 1 + d(sigma)_spot / (eps * d(sigma)/dk * d(ln S))
    denominator = forward_spot_elasticity * dvol_dk_early * d_ln_spot
    usable = np.abs(dvol_dk_early) >= minimum_abs_dvol_dk
    implied_r = np.full_like(k_early, np.nan)
    implied_r[usable] = 1.0 + dvol_spot[usable] / denominator[usable]

    vega = early.parameters.vega(k_early, early.forward, early.tau)
    weights = np.where(usable, vega, 0.0)
    if weights.sum() <= 0.0:
        raise ValueError(
            "Skew is too flat across the common range to identify R"
        )
    weighted_r = float(
        np.nansum(weights * np.nan_to_num(implied_r)) / weights.sum()
    )

    # Weighted least squares on the relation itself,
    #     dvol_spot = eps * (R - 1) * dvol_dk * d(lnS),
    # solved for R. Unlike the pointwise average this never divides by the
    # skew, so a flat patch contributes little information instead of a pole.
    design = forward_spot_elasticity * dvol_dk_early * d_ln_spot
    least_squares_denominator = float(np.sum(vega * design * design))
    least_squares_r = (
        1.0 + float(np.sum(vega * design * dvol_spot)) / least_squares_denominator
        if least_squares_denominator > 0.0
        else float("nan")
    )

    points = [
        MeasuredBackbonePoint(
            log_moneyness_early=float(k_early[i]),
            log_moneyness_late=float(k_late[i]),
            dvol_observed=float(dvol_observed[i]),
            dvol_time_component=float(dvol_time[i]),
            dvol_spot_component=float(dvol_spot[i]),
            dvol_dk_early=float(dvol_dk_early[i]),
            implied_stickiness_ratio=float(implied_r[i]),
            vega_weight=float(weights[i]),
        )
        for i in range(len(k_early))
    ]

    return MeasuredBackbone(
        expiration_timestamp=early.expiration_timestamp,
        d_ln_spot=d_ln_spot,
        d_tau=d_tau,
        tau_early=early.tau,
        tau_late=late.tau,
        vega_weighted_stickiness_ratio=weighted_r,
        least_squares_stickiness_ratio=least_squares_r,
        identification_note=note,
        points=points,
    )