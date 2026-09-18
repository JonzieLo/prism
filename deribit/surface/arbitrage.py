import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from deribit.pricing import Black76Model

from .results import (
    ArbitrageReport,
    ArbitrageViolation,
)
from .svi import SVIParameters


def svi_density_condition(
    parameters: SVIParameters,
    k: ArrayLike,
) -> NDArray[np.float64]:
    values = np.asarray(k, dtype=float)
    w = parameters.total_variance(values)
    w_prime = parameters.first_derivative(values)
    w_second = parameters.second_derivative(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            (1.0 - values * w_prime / (2.0 * w)) ** 2
            - 0.25 * w_prime * w_prime * (1.0 / w + 0.25)
            + 0.5 * w_second
        )


def check_svi_slice_arbitrage(
    parameters: SVIParameters,
    *,
    forward: float,
    tau: float,
    rate: float,
    observed_k_min: float,
    observed_k_max: float,
    grid_size: int = 1001,
    k_padding: float = 0.50,
    tolerance: float = 1e-10,
) -> ArbitrageReport:
    if forward <= 0.0 or tau <= 0.0:
        raise ValueError("forward and tau must be positive")
    if grid_size < 5:
        raise ValueError("grid_size must be at least 5")

    k_min = min(observed_k_min - k_padding, -1.0)
    k_max = max(observed_k_max + k_padding, 1.0)
    k_grid = np.linspace(k_min, k_max, grid_size)
    strikes = forward * np.exp(k_grid)
    total_variance = parameters.total_variance(k_grid)

    violations: list[ArbitrageViolation] = []
    minimum_variance = float(np.min(total_variance))
    positive_variance = minimum_variance >= -tolerance
    if not positive_variance:
        index = int(np.argmin(total_variance))
        violations.append(
            ArbitrageViolation(
                check="positive_variance",
                coordinate=float(k_grid[index]),
                value=minimum_variance,
                tolerance=tolerance,
                detail="Fitted total variance is negative",
            )
        )

    left_slope = parameters.left_wing_slope
    right_slope = parameters.right_wing_slope
    valid_wings = (
        0.0 <= left_slope < 2.0
        and 0.0 <= right_slope < 2.0
    )
    if not valid_wings:
        violations.append(
            ArbitrageViolation(
                check="wing_slopes",
                coordinate=math.nan,
                value=max(left_slope, right_slope),
                tolerance=2.0,
                detail=(
                    f"left={left_slope:.6g}, right={right_slope:.6g}; "
                    "each must lie in [0, 2)"
                ),
            )
        )

    density = svi_density_condition(parameters, k_grid)
    finite_density = density[np.isfinite(density)]
    minimum_density = (
        float(np.min(finite_density))
        if finite_density.size
        else -math.inf
    )
    density_ok = (
        finite_density.size == density.size
        and minimum_density >= -tolerance
    )
    if not density_ok:
        index = (
            int(np.nanargmin(density))
            if finite_density.size
            else 0
        )
        violations.append(
            ArbitrageViolation(
                check="density_condition",
                coordinate=float(k_grid[index]),
                value=minimum_density,
                tolerance=tolerance,
                detail="SVI g(k) is negative or non-finite",
            )
        )

    call_monotone = False
    call_convex = False
    prices_within_bounds = False
    minimum_convexity = -math.inf
    if positive_variance:
        vols = np.sqrt(np.maximum(total_variance, 0.0) / tau)
        model = Black76Model()
        prices = np.array(
            [
                model.price(
                    forward,
                    float(strike),
                    tau,
                    float(vol),
                    rate,
                    "call",
                )
                for strike, vol in zip(strikes, vols)
            ]
        )
        df = math.exp(-rate * tau)
        lower = df * np.maximum(forward - strikes, 0.0)
        upper = np.full_like(strikes, df * forward)
        prices_within_bounds = bool(
            np.all(prices >= lower - tolerance)
            and np.all(prices <= upper + tolerance)
        )
        if not prices_within_bounds:
            violations.append(
                ArbitrageViolation(
                    check="price_bounds",
                    coordinate=math.nan,
                    value=float(
                        max(
                            np.max(lower - prices),
                            np.max(prices - upper),
                        )
                    ),
                    tolerance=tolerance,
                    detail="Reconstructed call violates Black-76 bounds",
                )
            )

        call_differences = np.diff(prices)
        call_monotone = bool(np.all(call_differences <= tolerance))
        if not call_monotone:
            index = int(np.argmax(call_differences))
            violations.append(
                ArbitrageViolation(
                    check="call_monotonicity",
                    coordinate=float(strikes[index]),
                    value=float(call_differences[index]),
                    tolerance=tolerance,
                    detail="Call price increases with strike",
                )
            )

        slopes = np.diff(prices) / np.diff(strikes)
        convexity = np.diff(slopes)
        minimum_convexity = float(np.min(convexity))
        call_convex = minimum_convexity >= -tolerance
        if not call_convex:
            index = int(np.argmin(convexity)) + 1
            violations.append(
                ArbitrageViolation(
                    check="call_convexity",
                    coordinate=float(strikes[index]),
                    value=minimum_convexity,
                    tolerance=tolerance,
                    detail="Call-price slope decreases with strike",
                )
            )

    butterfly_free = (
        positive_variance
        and valid_wings
        and density_ok
        and call_monotone
        and call_convex
        and prices_within_bounds
    )
    return ArbitrageReport(
        butterfly_free=butterfly_free,
        positive_variance=positive_variance,
        valid_wing_slopes=valid_wings,
        call_monotone=call_monotone,
        call_convex=call_convex,
        prices_within_bounds=prices_within_bounds,
        minimum_total_variance=minimum_variance,
        minimum_density_condition=minimum_density,
        minimum_convexity_margin=minimum_convexity,
        violations=tuple(violations),
    )


def assert_butterfly_free(report: ArbitrageReport) -> None:
    if report.butterfly_free:
        return
    detail = "; ".join(
        f"{item.check}: {item.detail} (value={item.value:.6g})"
        for item in report.violations
    )
    raise AssertionError(f"SVI slice contains butterfly arbitrage: {detail}")
