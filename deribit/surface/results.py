from dataclasses import dataclass

from .svi import SVIParameters


@dataclass(frozen=True)
class SmileResidual:
    source_row_id: int
    instrument_name: str
    option_type: str
    strike: float
    log_moneyness: float
    market_total_variance: float
    fitted_total_variance: float
    total_variance_residual: float
    market_iv: float
    fitted_iv: float
    iv_residual: float
    market_price_usd: float
    fitted_price_usd: float
    price_residual_usd: float
    bid_usd: float
    ask_usd: float
    spread_normalized_residual: float | None


@dataclass(frozen=True)
class ArbitrageViolation:
    check: str
    coordinate: float
    value: float
    tolerance: float
    detail: str


@dataclass(frozen=True)
class ArbitrageReport:
    butterfly_free: bool
    positive_variance: bool
    valid_wing_slopes: bool
    call_monotone: bool
    call_convex: bool
    prices_within_bounds: bool
    minimum_total_variance: float
    minimum_density_condition: float
    minimum_convexity_margin: float
    violations: tuple[ArbitrageViolation, ...]


@dataclass(frozen=True)
class CalibratedSmile:
    snapshot_id: int
    underlying_index: str
    expiration_timestamp: int
    tau: float
    forward: float
    rate: float
    parameters: SVIParameters
    observation_count: int
    source_row_ids: tuple[int, ...]
    observed_k_min: float
    observed_k_max: float
    objective_value: float
    rmse_total_variance: float
    converged: bool
    optimizer_message: str
    starting_point_count: int
    residuals: tuple[SmileResidual, ...]
    arbitrage_report: ArbitrageReport
