import itertools
import math

import numpy as np
from scipy.optimize import minimize

from deribit.pricing import Black76Model

from .arbitrage import check_svi_slice_arbitrage
from .observations import SurfaceObservation
from .results import CalibratedSmile, SmileResidual
from .svi import SVIParameters


class SVICalibrationError(ValueError):
    pass


def _validate_observations(
    observations: list[SurfaceObservation],
) -> None:
    if len(observations) < 5:
        raise SVICalibrationError(
            "At least five observations are required for five-parameter SVI"
        )
    if len({item.snapshot_id for item in observations}) != 1:
        raise SVICalibrationError("Observations span multiple snapshots")
    if len(
        {
            (item.underlying_index, item.expiration_timestamp)
            for item in observations
        }
    ) != 1:
        raise SVICalibrationError("Observations span multiple expiries")

    for name, values in (
        ("forward", [item.forward for item in observations]),
        ("tau", [item.tau for item in observations]),
        ("rate", [item.rate for item in observations]),
    ):
        if not np.allclose(values, values[0], rtol=1e-12, atol=1e-12):
            raise SVICalibrationError(f"Inconsistent {name} within expiry")


def _parameter_vector(parameters: SVIParameters) -> np.ndarray:
    return np.array(
        [
            parameters.a,
            parameters.b,
            parameters.rho,
            parameters.m,
            parameters.eta,
        ],
        dtype=float,
    )


def _parameters(values: np.ndarray) -> SVIParameters:
    return SVIParameters(
        a=float(values[0]),
        b=float(values[1]),
        rho=float(values[2]),
        m=float(values[3]),
        eta=float(values[4]),
    )


def _starting_points(
    k: np.ndarray,
    market_w: np.ndarray,
    number_of_starts: int,
) -> list[SVIParameters]:
    k_span = max(float(np.ptp(k)), 0.10)
    minimum_index = int(np.argmin(market_w))
    m_candidates = [float(k[minimum_index]), 0.0]
    eta_candidates = [
        max(0.05, 0.20 * k_span),
        max(0.10, 0.50 * k_span),
    ]
    rho_candidates = [-0.70, -0.30, 0.0, 0.30]
    slope_scale = max(
        float(np.ptp(market_w)) / k_span,
        0.01,
    )
    b_candidates = [
        max(0.01, 0.50 * slope_scale),
        max(0.02, slope_scale),
    ]

    starts: list[SVIParameters] = []
    for m, eta, rho, b in itertools.product(
        m_candidates,
        eta_candidates,
        rho_candidates,
        b_candidates,
    ):
        a = max(
            float(np.min(market_w))
            - b * eta * math.sqrt(1.0 - rho * rho),
            -1.0,
        )
        starts.append(SVIParameters(a, b, rho, m, eta))
        if len(starts) >= number_of_starts:
            break
    return starts


def calibrate_svi_slice(
    observations: list[SurfaceObservation],
    *,
    number_of_starts: int = 16,
    arbitrage_grid_size: int = 1001,
    require_butterfly_free: bool = True,
) -> CalibratedSmile:
    _validate_observations(observations)
    if number_of_starts < 1:
        raise ValueError("number_of_starts must be positive")

    ordered = sorted(observations, key=lambda item: item.log_moneyness)
    k = np.array([item.log_moneyness for item in ordered])
    market_w = np.array([item.total_variance for item in ordered])
    if not np.all(np.isfinite(market_w)) or np.any(market_w <= 0.0):
        raise SVICalibrationError(
            "Market total variance must be finite and positive"
        )

    k_span = max(float(np.ptp(k)), 0.10)
    w_scale = max(float(np.max(market_w)), 0.01)
    bounds = (
        (-2.0 * w_scale, 2.0 * w_scale),
        (1e-8, 5.0),
        (-0.999, 0.999),
        (float(np.min(k) - k_span), float(np.max(k) + k_span)),
        (1e-5, max(2.0, 2.0 * k_span)),
    )

    def objective(values: np.ndarray) -> float:
        fitted = _parameters(values).total_variance(k)
        residual = fitted - market_w
        return float(np.dot(residual, residual))

    def minimum_variance_constraint(values: np.ndarray) -> float:
        return _parameters(values).minimum_total_variance

    candidates = []
    starts = _starting_points(k, market_w, number_of_starts)
    for start in starts:
        result = minimize(
            objective,
            _parameter_vector(start),
            method="SLSQP",
            bounds=bounds,
            constraints=(
                {
                    "type": "ineq",
                    "fun": minimum_variance_constraint,
                },
            ),
            options={
                "ftol": 1e-14,
                "maxiter": 2_000,
            },
        )
        if not result.success or not np.isfinite(result.fun):
            continue
        parameters = _parameters(result.x)
        try:
            parameters.validate()
        except ValueError:
            continue
        arbitrage = check_svi_slice_arbitrage(
            parameters,
            forward=ordered[0].forward,
            tau=ordered[0].tau,
            rate=ordered[0].rate,
            observed_k_min=float(np.min(k)),
            observed_k_max=float(np.max(k)),
            grid_size=arbitrage_grid_size,
        )
        if require_butterfly_free and not arbitrage.butterfly_free:
            continue
        candidates.append((float(result.fun), result, parameters, arbitrage))

    if not candidates:
        qualifier = " butterfly-free" if require_butterfly_free else ""
        raise SVICalibrationError(
            f"No converged{qualifier} SVI fit across {len(starts)} starts"
        )

    objective_value, optimizer, parameters, arbitrage = min(
        candidates,
        key=lambda item: item[0],
    )
    fitted_w = parameters.total_variance(k)
    fitted_iv = np.sqrt(fitted_w / ordered[0].tau)
    model = Black76Model()
    residuals: list[SmileResidual] = []
    for item, model_w, model_iv in zip(ordered, fitted_w, fitted_iv):
        fitted_price = model.price(
            item.forward,
            item.strike,
            item.tau,
            float(model_iv),
            item.rate,
            item.option_type,
        )
        half_spread = 0.5 * (item.ask_usd - item.bid_usd)
        price_residual = item.mid_usd - fitted_price
        normalized = (
            price_residual / half_spread
            if half_spread > 0.0
            else None
        )
        residuals.append(
            SmileResidual(
                source_row_id=item.source_row_id,
                instrument_name=item.instrument_name,
                option_type=item.option_type,
                strike=item.strike,
                log_moneyness=item.log_moneyness,
                market_total_variance=item.total_variance,
                fitted_total_variance=float(model_w),
                total_variance_residual=(
                    item.total_variance - float(model_w)
                ),
                market_iv=item.mid_iv,
                fitted_iv=float(model_iv),
                iv_residual=item.mid_iv - float(model_iv),
                market_price_usd=item.mid_usd,
                fitted_price_usd=fitted_price,
                price_residual_usd=price_residual,
                bid_usd=item.bid_usd,
                ask_usd=item.ask_usd,
                spread_normalized_residual=normalized,
            )
        )

    first = ordered[0]
    return CalibratedSmile(
        snapshot_id=first.snapshot_id,
        underlying_index=first.underlying_index,
        expiration_timestamp=first.expiration_timestamp,
        tau=first.tau,
        forward=first.forward,
        rate=first.rate,
        parameters=parameters,
        observation_count=len(ordered),
        source_row_ids=tuple(item.source_row_id for item in ordered),
        observed_k_min=float(np.min(k)),
        observed_k_max=float(np.max(k)),
        objective_value=objective_value,
        rmse_total_variance=math.sqrt(objective_value / len(ordered)),
        converged=bool(optimizer.success),
        optimizer_message=str(optimizer.message),
        starting_point_count=len(starts),
        residuals=tuple(residuals),
        arbitrage_report=arbitrage,
    )
