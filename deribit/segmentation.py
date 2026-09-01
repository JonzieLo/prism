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
        bucket_pair_count = len(items)
        if bucket_pair_count == 0:
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
            val
            for item in items
            if (
                val := observed_pair_sum(
                    item.pair.call.open_interest, item.pair.put.open_interest
                )
            )
            is not None
        ]
        volumes = [
            val
            for item in items
            if (
                val := observed_pair_sum(
                    item.pair.call.volume, item.pair.put.volume
                )
            )
            is not None
        ]

        metrics.append(
            MoneynessBucketMetrics(
                band=band,
                pair_count=bucket_pair_count,
                eligible_pair_count=eligible_count,
                midpoint_eligible_fraction=eligible_count / bucket_pair_count,
                missing_bid_fraction=missing_bid_count / bucket_pair_count,
                unusable_bid_fraction=unusable_bid_count / bucket_pair_count,
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


@dataclass(frozen=True)
class LiquidityGroupMetrics:
    group_name: str
    pair_count: int
    midpoint_eligible_fraction: float
    median_relative_spread: float | None
    median_abs_parity_residual: float | None
    median_synthetic_interval_width: float | None

def build_liquidity_segmentation(
    evaluated_pairs: Sequence[EvaluatedPair],
    expiry_forwards: Sequence[ExpiryForward],
) -> tuple[LiquidityGroupMetrics, ...]:
    """Segment all pairs cross-sectionally by snapshot-relative liquidity groups with residuals."""
    if not evaluated_pairs:
        return ()

    forward_by_expiry = {
        (
            expiry.underlying_index,
            expiry.expiration_timestamp,
        ): expiry.implied_forward
        for expiry in expiry_forwards
    }

    missing_volume = []
    zero_vol = []
    pos_vol = []
    for item in evaluated_pairs:
        vol = observed_pair_sum(item.pair.call.volume, item.pair.put.volume)
        if vol is None:
            missing_volume.append(item)
        elif vol == 0.0:
            zero_vol.append(item)
        else:
            pos_vol.append((vol, item))

    groups = []
    if missing_volume:
        groups.append(("Missing Volume Data", missing_volume))
    if zero_vol:
        groups.append(("Zero Observed Volume", zero_vol))

    if pos_vol:
        pos_vol.sort(key=lambda x: x[0])
        vols = [x[0] for x in pos_vol]
        q25, q75 = np.quantile(vols, [0.25, 0.75])

        q1 = [item for v, item in pos_vol if v <= q25]
        q2_3 = [item for v, item in pos_vol if q25 < v <= q75]
        q4 = [item for v, item in pos_vol if v > q75]

        groups.extend([
            ("Lower Quartile Vol", q1),
            ("Middle 50% Vol", q2_3),
            ("Upper Quartile Vol", q4),
        ])

    metrics = []
    for name, items in groups:
        count = len(items)
        if count == 0:
            continue
        eligible = [item for item in items if item.diagnostic_eligible]

        spreads = [
            s for item in items
            for s in (relative_spread(item.pair.call), relative_spread(item.pair.put))
            if s is not None
        ]

        residuals = []
        for item in eligible:
            key = (item.pair.underlying_index, item.pair.expiration_timestamp)
            ref_fwd = forward_by_expiry.get(key)
            if ref_fwd is not None and item.point.forward_mid is not None:
                residuals.append(abs(item.point.forward_mid - ref_fwd))

        widths = [
            item.point.synthetic_buy_forward - item.point.synthetic_sell_forward
            for item in items
            if item.point.synthetic_buy_forward is not None and item.point.synthetic_sell_forward is not None
        ]

        metrics.append(
            LiquidityGroupMetrics(
                group_name=name,
                pair_count=count,
                midpoint_eligible_fraction=len(eligible) / count,
                median_relative_spread=float(np.median(spreads)) if spreads else None,
                median_abs_parity_residual=float(np.median(residuals)) if residuals else None,
                median_synthetic_interval_width=float(np.median(widths)) if widths else None,
            )
        )

    return tuple(metrics)