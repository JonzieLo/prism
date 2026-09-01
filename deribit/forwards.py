import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict
from .chain import OptionQuote, FutureQuote

@dataclass(frozen=True)
class OptionPair:
    expiration_timestamp: int
    underlying_index: str
    strike: float
    call: OptionQuote
    put: OptionQuote


@dataclass(frozen=True)
class ParityPoint:
    expiration_timestamp: int
    underlying_index: str
    strike: float
    forward_mid: float | None
    synthetic_buy_forward: float | None
    synthetic_sell_forward: float | None
    synthetic_buy_max_size: float | None
    synthetic_sell_max_size: float | None


@dataclass(frozen=True)
class ExpiryForward:
    expiration_timestamp: int
    underlying_index: str
    implied_forward: float
    dispersion_mad: float
    dispersion_iqr: float
    pair_count: int
    best_synthetic_buy: float | None
    best_synthetic_buy_strike: float | None
    best_synthetic_sell: float | None
    best_synthetic_sell_strike: float | None


@dataclass(frozen=True)
class BasisComparison:
    expiration_timestamp: int
    underlying_index: str
    implied_forward: float
    future_bid: float | None
    future_ask: float | None
    future_mark: float | None
    basis_usd: float | None
    basis_bps: float | None
    best_synthetic_buy: float | None
    best_synthetic_sell: float | None
    top_of_book_cross_direction: str | None
    status: str


@dataclass(frozen=True)
class PairingIssue:
    key: tuple[str | None, int, float]
    reason: str
    source_row_ids: tuple[int, ...]

@dataclass(frozen=True)
class BasisStatus(str, Enum):
    OK = "ok"
    MISSING_FUTURE = "missing_future"
    INVALID_FUTURE_MARK = "invalid_future_mark"
    CROSSED_FUTURE = "crossed_future"
    INVALID_FUTURE_BOOK = "invalid_future_book"
    PRICE_CROSS = "price_cross"
    BOTH_DIRECTIONS_CROSS = "both_directions_cross"

def pair_calls_and_puts(
        quotes: list[OptionQuote],
) -> tuple[list[OptionPair], list[PairingIssue]]: ## later implement QuoteIssue class
    """
    ---- Pairing -----
    Requirements:
     - same expiry
     - same strike
     - same underlying_index
     - same settlement currency, contract size
     - one call, one put
     - preserve unpaired calls and puts an QuoteIssue
     - does *not* overwrite duplicate rows
    """
    groups: dict[
        tuple[str | None, int, float],
        list[OptionQuote],
    ] = defaultdict(list)

    for quote in quotes:
        key = (
            quote.underlying_index,
            quote.expiration_timestamp,
            quote.strike,
        )
        groups[key].append(quote)

    pairs: list[OptionPair] = []
    issues: list[PairingIssue] = []

    for key, group in groups.items():
        calls = [
            quote
            for quote in group
            if quote.option_type == "call"
        ]
        puts = [
            quote
            for quote in group
            if quote.option_type == "put"
        ]
        source_ids = tuple(
            sorted(quote.source_row_id for quote in group)
        )

        reasons: list[str] = []

        if len(calls) == 0:
            reasons.append("missing_call")
        elif len(calls) > 1:
            reasons.append("duplicate_call")

        if len(puts) == 0:
            reasons.append("missing_put")
        elif len(puts) > 1:
            reasons.append("duplicate_put")

        if reasons:
            issues.extend(
                PairingIssue(
                    key=key,
                    reason=reason,
                    source_row_ids=source_ids,
                )
                for reason in reasons
            )
            continue

        call = calls[0]
        put = puts[0]

        if call.settlement_currency != put.settlement_currency:
            issues.append(
                PairingIssue(
                    key=key,
                    reason="settlement_currency_mismatch",
                    source_row_ids=source_ids,
                )
            )
            continue

        if call.contract_size != put.contract_size:
            issues.append(
                PairingIssue(
                    key=key,
                    reason="contract_size_mismatch",
                    source_row_ids=source_ids,
                )
            )
            continue

        if key[0] is None:
            issues.append(
                PairingIssue(
                    key=key,
                    reason="missing_underlying_index",
                    source_row_ids=source_ids,
                )
            )
            continue

        pairs.append(
            OptionPair(
                expiration_timestamp=key[1],
                underlying_index=key[0],
                strike=key[2],
                call=call,
                put=put,
            )
        )

    return pairs, issues

def inverse_forward_mid(pair: OptionPair) -> float | None:
    "Midpoint Parity"
    call_mid = pair.call.mid_coin
    put_mid = pair.put.mid_coin

    if call_mid is None or put_mid is None:
        return None
    if not math.isfinite(call_mid) or not math.isfinite(put_mid):
        return None
    
    denom = 1.0 - call_mid + put_mid
    if denom <= 0.0 or not math.isfinite(denom):
        raise ValueError("Invalid inverse parity denominator")
    return pair.strike/denom

#Independent execution sides
### assert synthetic_sell_forward <= forward_mid <= synthetic_buy_forward
def synthetic_buy_forward(pair: OptionPair) -> float | None:
    if (
        pair.call.ask_coin is None
        or pair.put.bid_coin is None
        or not math.isfinite(pair.call.ask_coin)
        or not math.isfinite(pair.put.bid_coin)
        or pair.call.ask_coin <= 0.0
        or pair.put.bid_coin <= 0.0
    ):
        return None
    denom = (
        1.0 - pair.call.ask_coin + pair.put.bid_coin
    )
    if denom <= 0.0:
        raise ValueError("Invalid synthetic-buy denominator")
    return pair.strike/denom

def synthetic_sell_forward(pair: OptionPair) -> float | None:
    if (
        pair.call.bid_coin is None
        or pair.put.ask_coin is None
        or not math.isfinite(pair.call.bid_coin)
        or not math.isfinite(pair.put.ask_coin)
        or pair.call.bid_coin <= 0.0
        or pair.put.ask_coin <= 0.0
    ):
        return None
    denom = (
        1.0 - pair.call.bid_coin + pair.put.ask_coin
    )
    if denom <= 0.0:
        raise ValueError("Invalid synthetic-sell denominator")
    return pair.strike/denom

# Matched Size Helper Functions
def matched_size(
    first: float | None,
    second: float | None,
) -> float | None:
    if (
        first is None
        or second is None
        or first <= 0.0
        or second <= 0.0
    ):
        return None
    return min(first, second)


def parity_point(pair: OptionPair) -> ParityPoint:
    "Parity Point Constructor"
    return ParityPoint(
        expiration_timestamp=pair.expiration_timestamp,
        underlying_index=pair.underlying_index,
        strike=pair.strike,
        forward_mid=inverse_forward_mid(pair),
        synthetic_buy_forward=synthetic_buy_forward(pair),
        synthetic_sell_forward=synthetic_sell_forward(pair),
        synthetic_buy_max_size=matched_size(
            pair.call.ask_amount,
            pair.put.bid_amount,
        ),
        synthetic_sell_max_size=matched_size(
            pair.call.bid_amount,
            pair.put.ask_amount,
        ),
    )

#Expiry aggregation
def aggregate_expiry_forward(
    diagnostic_points: list[ParityPoint],
    buy_points: list[ParityPoint],
    sell_points: list[ParityPoint],
) -> ExpiryForward:
    """
    Generates aggregated implied forward.
    
    1. Computes the median forward_mid across diagnostic_points as the implied forward.
    2. Calculates MAD (Median Absolute Deviation) and IQR dispersion metrics.
    3. Finds the best (minimum) synthetic buy forward and best (maximum) synthetic sell forward.
    """
    diagnostic = [p for p in diagnostic_points if p.forward_mid is not None]
    buy = [p for p in buy_points if p.synthetic_buy_forward is not None]
    sell = [p for p in sell_points if p.synthetic_sell_forward is not None]

    if not diagnostic:
        raise ValueError("Expiry has no diagnostic-eligible parity points")

    keys = {(p.underlying_index, p.expiration_timestamp) for p in [*diagnostic, *buy, *sell]}
    if len(keys) != 1:
        raise ValueError("Cannot aggregate multiple expiries or underlyings")

    values = np.array([p.forward_mid for p in diagnostic], dtype=float)
    implied = float(np.median(values))
    mad = float(np.median(np.abs(values - implied)))
    q25, q75 = np.quantile(values, [0.25, 0.75])

    best_buy = min(buy, key=lambda p: p.synthetic_buy_forward, default=None)
    best_sell = max(sell, key=lambda p: p.synthetic_sell_forward, default=None)

    underlying_index, expiration = keys.pop()

    return ExpiryForward(
        expiration_timestamp=expiration,
        underlying_index=underlying_index,
        implied_forward=implied,
        dispersion_mad=mad,
        dispersion_iqr=float(q75 - q25),
        pair_count=len(diagnostic),
        best_synthetic_buy=best_buy.synthetic_buy_forward if best_buy else None,
        best_synthetic_buy_strike=best_buy.strike if best_buy else None,
        best_synthetic_sell=best_sell.synthetic_sell_forward if best_sell else None,
        best_synthetic_sell_strike=best_sell.strike if best_sell else None,
    )


def compare_with_future(
        expiry_forward: ExpiryForward,
        future: FutureQuote | None,
) -> BasisComparison:
    """Compares aggregated implied forward against the traded future."""

    if future is None:
        return BasisComparison(
            expiration_timestamp=expiry_forward.expiration_timestamp,
            underlying_index=expiry_forward.underlying_index,
            implied_forward=expiry_forward.implied_forward,
            future_bid=None,
            future_ask=None,
            future_mark=None,
            basis_usd=None,
            basis_bps=None,
            best_synthetic_buy=expiry_forward.best_synthetic_buy,
            best_synthetic_sell=expiry_forward.best_synthetic_sell,
            top_of_book_cross_direction=None,
            status=BasisStatus.MISSING_FUTURE.value,
        )

    future_book_valid = True
    if future.bid is not None:
        if not math.isfinite(future.bid) or future.bid <= 0.0:
            future_book_valid = False
    if future.ask is not None:
        if not math.isfinite(future.ask) or future.ask <= 0.0:
            future_book_valid = False
    if (
        future.bid is not None
        and future.ask is not None
        and future_book_valid
        and future.bid > future.ask
    ):
        future_book_valid = False

    if not future_book_valid:
        has_valid_mark = (
            future.mark is not None
            and math.isfinite(future.mark)
            and future.mark > 0.0
        )
        basis_usd = (expiry_forward.implied_forward - future.mark) if has_valid_mark else None
        basis_bps = (10_000.0 * (expiry_forward.implied_forward / future.mark - 1.0)) if has_valid_mark else None

        return BasisComparison(
            expiration_timestamp=expiry_forward.expiration_timestamp,
            underlying_index=expiry_forward.underlying_index,
            implied_forward=expiry_forward.implied_forward,
            future_bid=future.bid,
            future_ask=future.ask,
            future_mark=future.mark,
            basis_usd=basis_usd,
            basis_bps=basis_bps,
            best_synthetic_buy=expiry_forward.best_synthetic_buy,
            best_synthetic_sell=expiry_forward.best_synthetic_sell,
            top_of_book_cross_direction=None,
            status=BasisStatus.INVALID_FUTURE_BOOK.value,
        )

    buy_cross = (
        future.bid is not None
        and expiry_forward.best_synthetic_buy is not None
        and future.bid > expiry_forward.best_synthetic_buy
    )

    sell_cross = (
        future.ask is not None
        and expiry_forward.best_synthetic_sell is not None
        and expiry_forward.best_synthetic_sell > future.ask
    )

    if buy_cross and sell_cross:
        cross_direction = "both_directions_cross"
        status = BasisStatus.BOTH_DIRECTIONS_CROSS.value
    elif buy_cross:
        cross_direction = "buy_synthetic_sell_future"
        status = BasisStatus.PRICE_CROSS.value
    elif sell_cross:
        cross_direction = "sell_synthetic_buy_future"
        status = BasisStatus.PRICE_CROSS.value
    else:
        cross_direction = None
        status = BasisStatus.OK.value

    # Midpoint Basis
    has_valid_mark = (
        future.mark is not None 
        and math.isfinite(future.mark) 
        and future.mark > 0.0
    )

    if has_valid_mark:
        basis_usd = expiry_forward.implied_forward - future.mark
        basis_bps = 10_000.0 * (expiry_forward.implied_forward / future.mark - 1.0)
    else:
        basis_usd = None
        basis_bps = None
        if status == BasisStatus.OK.value:
            status = BasisStatus.INVALID_FUTURE_MARK.value

    return BasisComparison(
        expiration_timestamp=expiry_forward.expiration_timestamp,
        underlying_index=expiry_forward.underlying_index,
        implied_forward=expiry_forward.implied_forward,
        future_bid=future.bid,
        future_ask=future.ask,
        future_mark=future.mark,
        basis_usd=basis_usd,
        basis_bps=basis_bps,
        best_synthetic_buy=expiry_forward.best_synthetic_buy,
        best_synthetic_sell=expiry_forward.best_synthetic_sell,
        top_of_book_cross_direction=cross_direction,
        status=status,
    )