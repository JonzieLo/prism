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
CALENDAR_ARBITRAGE_RELATIVE_TOLERANCE = -1e-8
DEFAULT_MINIMUM_OBSERVATIONS = 10


def fit_snapshot_term_structure(
    observations_by_expiry: dict[int, list[SurfaceObservation]],
    *,
    number_of_starts: int = 16,
    min_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
    require_butterfly_free: bool = True,
) -> list[CalibratedSmile]:
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
        raise ValueError("Calibrated smiles do not share an observed log-moneyness range")
    return np.linspace(k_min, k_max, n_points)


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



def dw_dtau_between(
    near: CalibratedSmile,
    far: CalibratedSmile,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """d(w)/d(tau) at fixed k"""
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

    At a fixed strike K:
        d(sigma_K) = (d sigma / d ln S) * d(ln S)
                   + (d sigma / d tau)  * d(tau)
                   + residual
    """
    if late.expiration_timestamp != early.expiration_timestamp:
        raise ValueError("Both smiles must be the same expiry")
    if late.tau >= early.tau:
        raise ValueError("`late` must be the later snapshot (smaller tau)")

    d_ln_spot = float(np.log(late.forward / early.forward))
    if abs(d_ln_spot) < 1e-6:
        raise ValueError("Forwards are unchanged between snapshots; spot component is not identified")
    d_tau = late.tau - early.tau

    k_early = np.linspace(
        max(early.observed_k_min, late.observed_k_min + d_ln_spot),
        min(early.observed_k_max, late.observed_k_max + d_ln_spot),
        n_points,
    )
    if k_early[0] >= k_early[-1]:
        raise ValueError("Snapshots share no common fixed-strike range")

    k_late = k_early - forward_spot_elasticity * d_ln_spot

    sigma_early = early.parameters.implied_vol(k_early, early.tau)
    sigma_late = late.parameters.implied_vol(k_late, late.tau)
    dvol_observed = sigma_late - sigma_early

    w_early = early.parameters.total_variance(k_early)
    if dw_dtau is None:
        dw_dtau_array = np.zeros_like(k_early) #assume - dw_dtau
    elif callable(dw_dtau):
        dw_dtau_array = np.asarray(dw_dtau(k_early), dtype=float) #Time component removed 
        if dw_dtau_array.shape != k_early.shape:
            raise ValueError(f"dw_dtau callable returned shape {dw_dtau_array.shape}, expected {k_early.shape}")
    else:
        dw_dtau_array = np.broadcast_to(np.asarray(dw_dtau, dtype=float), k_early.shape) #Time component removed using positional dw_dtau

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

    # dvol_spot = eps * (R - 1) * dvol_dk * d(lnS),
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
        points=points,
    )