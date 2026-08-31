import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .chain import OptionQuote
from .forwards import ExpiryForward
from .hygiene import IssueCode, EvaluatedPair

INNER = 0.05
OUTER = 0.20

class MoneynessBand(str, Enum):
    FAR_BELOW_FORWARD = "far_below_forward"
    BELOW_FORWARD = "below_forward"
    NEAR_FORWARD = "near_forward"
    ABOVE_FORWARD = "above_forward"
    FAR_ABOVE_FORWARD = "far_above_forward"


ORDERED_BANDS = tuple(MoneynessBand)

UNUSABLE_BID_CODES = {
    IssueCode.MISSING_BID,
    IssueCode.NON_FINITE_BID,
    IssueCode.ZERO_BID,
}

@dataclass(frozen=True)
class MoneynessBucketMetrics:
    band: MoneynessBand
    pair_count: int
    eligible_pair_count: int
    midpoint_eligible_fraction: float | None
    missing_bid_fraction: float | None
    unusable_bid_fraction: float | None
    call_spread_count: int
    median_call_relative_spread: float | None
    put_spread_count: int
    median_put_relative_spread: float | None
    residual_count: int
    median_abs_internal_residual_usd: float | None
    internal_residual_mad_usd: float | None
    liquidity_count: int
    median_open_interest: float | None
    median_volume: float | None

def log_moneyness(strike: float, forward: float) -> float:
    if (
        not math.isfinite(strike)
        or not math.isfinite(forward)
        or strike <= 0.0
        or forward <= 0.0
    ):
        raise ValueError("Strike and forward must be finite and positive")
    return math.log(strike / forward)


def moneyness_band(value: float) -> MoneynessBand:
    if not math.isfinite(value):
        raise ValueError("Log-moneyness must be finite")
    if value < -OUTER:
        return MoneynessBand.FAR_BELOW_FORWARD
    if value < -INNER:
        return MoneynessBand.BELOW_FORWARD
    if value <= INNER:
        return MoneynessBand.NEAR_FORWARD
    if value <= OUTER:
        return MoneynessBand.ABOVE_FORWARD
    return MoneynessBand.FAR_ABOVE_FORWARD


def relative_spread(quote: OptionQuote) -> float | None:
    "Calculate relative spread (ask - bid) / mid"
    bid = quote.bid_coin
    ask = quote.ask_coin
    if bid is None or ask is None:
        return None
    if not math.isfinite(bid) or not math.isfinite(ask):
        return None
    if bid <= 0.0 or ask <= 0.0 or bid > ask:
        return None
    midpoint = 0.5 * (bid + ask)
    return (ask - bid) / midpoint


def observed_pair_sum(
    first: float | None,
    second: float | None,
) -> float | None:
    "Sum two option values if both exist and are finite, returning None otherwise."
    values = [
        value
        for value in (first, second)
        if value is not None and math.isfinite(value)
    ]
    return sum(values) if len(values) == 2 else None


def empty_bucket_metrics(band: MoneynessBand) -> MoneynessBucketMetrics:
    "Return metrics instance for empty bands."
    return MoneynessBucketMetrics(
        band=band,
        pair_count=0,
        eligible_pair_count=0,
        midpoint_eligible_fraction=None,
        missing_bid_fraction=None,
        unusable_bid_fraction=None,
        call_spread_count=0,
        median_call_relative_spread=None,
        put_spread_count=0,
        median_put_relative_spread=None,
        residual_count=0,
        median_abs_internal_residual_usd=None,
        internal_residual_mad_usd=None,
        liquidity_count=0,
        median_open_interest=None,
        median_volume=None,
    )


def build_moneyness_segmentation(
    evaluated_pairs: Sequence[EvaluatedPair],
    expiry_forward: ExpiryForward,
) -> tuple[MoneynessBucketMetrics, ...]:
    """Segment an expiry's evaluated pairs into log-moneyness bands and summarize quote metrics."""
    reference_forward = expiry_forward.implied_forward
    if not math.isfinite(reference_forward) or reference_forward <= 0.0:
        raise ValueError("Options-implied forward must be finite and positive")

    key = (
        expiry_forward.underlying_index,
        expiry_forward.expiration_timestamp,
    )
    matching = [
        item
        for item in evaluated_pairs
        if (
            item.pair.underlying_index,
            item.pair.expiration_timestamp,
        )
        == key
    ]

    if not matching:
        raise ValueError(
            "No evaluated pairs match the requested expiry forward"
        )

    grouped: dict[MoneynessBand, list[EvaluatedPair]] = defaultdict(list)
    for item in matching:
        m_val = log_moneyness(item.pair.strike, reference_forward)
        grouped[moneyness_band(m_val)].append(item)

    metrics = []
    for band in ORDERED_BANDS:
        items = grouped[band]
        total = len(items)
        if total == 0:
            metrics.append(empty_bucket_metrics(band))
            continue

        eligible = [item for item in items if item.diagnostic_eligible]
        eligible_count = len(eligible)

        # Hygiene Issue Code Counts
        missing_bid_count = sum(
            any(issue.code == IssueCode.MISSING_BID for issue in item.issues)
            for item in items
        )
        unusable_bid_count = sum(
            any(issue.code in UNUSABLE_BID_CODES for issue in item.issues)
            for item in items
        )

        # Spreads
        call_spreads = [
            v for item in items if (v := relative_spread(item.pair.call)) is not None
        ]
        put_spreads = [
            v for item in items if (v := relative_spread(item.pair.put)) is not None
        ]

        # Signed Residuals
        signed_residuals = np.array(
            [
                item.point.forward_mid - reference_forward
                for item in eligible
                if item.point.forward_mid is not None
            ],
            dtype=float,
        )

        if signed_residuals.size > 0:
            median_abs_res = float(np.median(np.abs(signed_residuals)))
            res_median = float(np.median(signed_residuals))
            res_mad = float(np.median(np.abs(signed_residuals - res_median)))
        else:
            median_abs_res = None
            res_mad = None

        # Liquidity
        open_interests = [
            total
            for item in items
            if (
                total := observed_pair_sum(
                    item.pair.call.open_interest, item.pair.put.open_interest
                )
            )
            is not None
        ]
        volumes = [
            total
            for item in items
            if (
                total := observed_pair_sum(
                    item.pair.call.volume, item.pair.put.volume
                )
            )
            is not None
        ]

        metrics.append(
            MoneynessBucketMetrics(
                band=band,
                pair_count=total,
                eligible_pair_count=eligible_count,
                midpoint_eligible_fraction=eligible_count / total,
                missing_bid_fraction=missing_bid_count / total,
                unusable_bid_fraction=unusable_bid_count / total,
                call_spread_count=len(call_spreads),
                median_call_relative_spread=float(np.median(call_spreads)) if call_spreads else None,
                put_spread_count=len(put_spreads),
                median_put_relative_spread=float(np.median(put_spreads)) if put_spreads else None,
                residual_count=int(signed_residuals.size),
                median_abs_internal_residual_usd=median_abs_res,
                internal_residual_mad_usd=res_mad,
                liquidity_count=len(open_interests),
                median_open_interest=float(np.median(open_interests)) if open_interests else None,
                median_volume=float(np.median(volumes)) if volumes else None,
            )
        )

    return tuple(metrics)


def format_optional_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def format_optional_float(value: float | None, precision: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{precision}f}"


def format_moneyness_table(
    metrics: Sequence[MoneynessBucketMetrics],
) -> str:
    headers = (
        f"{'Band':<18} {'Pairs':<6} {'Elig%':<8} {'MissBid%':<9} {'UnuseBid%':<10} "
        f"{'CallSprd':<9} {'PutSprd':<9} {'MedAbsRes':<10} {'ResMAD':<9} "
        f"{'MedOI':<8} {'MedVol':<8}"
    )
    lines = [
        "--- MONEYNESS SEGMENTATION ---",
        headers,
        "-" * len(headers),
    ]

    for m in metrics:
        lines.append(
            f"{m.band.value:<18} "
            f"{m.pair_count:<6} "
            f"{format_optional_percent(m.midpoint_eligible_fraction):<8} "
            f"{format_optional_percent(m.missing_bid_fraction):<9} "
            f"{format_optional_percent(m.unusable_bid_fraction):<10} "
            f"{format_optional_float(m.median_call_relative_spread):<9} "
            f"{format_optional_float(m.median_put_relative_spread):<9} "
            f"{format_optional_float(m.median_abs_internal_residual_usd, 2):<10} "
            f"{format_optional_float(m.internal_residual_mad_usd, 2):<9} "
            f"{format_optional_float(m.median_open_interest, 0):<8} "
            f"{format_optional_float(m.median_volume, 0):<8}"
        )

    return "\n".join(lines)


