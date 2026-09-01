from dataclasses import dataclass
from enum import Enum

class SurfaceExclusionCode(str, Enum):
    MISSING_EXPIRY_CONTEXT = "missing_expiry_context"
    DUPLICATE_OPTION_LEG = "duplicate_option_leg"
    MISSING_CANONICAL_OTM_LEG = "missing_canonical_otm_leg"
    INVALID_BID_ASK = "invalid_bid_ask"
    CROSSED_BOOK = "crossed_book"
    SPREAD_POLICY = "spread_policy"
    BELOW_INTRINSIC = "below_intrinsic"
    AT_OR_ABOVE_CEILING = "at_or_above_ceiling"
    MID_IV_FAILURE = "mid_iv_failure"
    MID_REPRICING_FAILURE = "mid_repricing_failure"
    INSUFFICIENT_EXPIRY_POINTS = "insufficient_expiry_points"

@dataclass(frozen=True)
class SurfaceExclusion:
    source_row_id: int | None
    instrument_name: str | None
    underlying_index: str
    expiration_timestamp: int
    strike: float
    reason: SurfaceExclusionCode
    detail: str

@dataclass(frozen=True)
class SurfaceObservation:
    source_row_id: int
    instrument_name: str
    underlying_index: str
    expiration_timestamp: int
    index_price: float
    forward: float
    strike: float
    tau: float
    rate: float
    log_moneyness: float  # k = ln(K/F)
    option_type: str
    bid_coin: float
    ask_coin: float
    mid_coin: float
    bid_usd: float
    ask_usd: float
    mid_usd: float
    bid_iv: float | None
    mid_iv: float
    ask_iv: float | None
    total_variance: float  # w = mid_iv^2 * tau
    relative_spread: float
    open_interest: float | None
    volume: float | None

@dataclass(frozen=True)
class SurfaceObservationResult:
    observations: tuple[SurfaceObservation, ...]
    exclusions: tuple[SurfaceExclusion, ...]