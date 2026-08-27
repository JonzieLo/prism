import numpy as np
import math
from dataclasses import dataclass
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
    for quote in quotes:
        key = (
            quote.underlying_index,
            quote.expiration_timestamp,
            quote.strike,
        )
    ...

def inverse_forward_mid(pair: OptionPair) -> float | None:
    "Midpoint Parity"
    call_mid = pair.call.mid_coin
    put_mid = pair.put.mid_coin

    if call_mid is None or put_mid is None:
        return None

    denom = 1.0 - call_mid + put_mid
    if denom <= 0.0:
        raise ValueError("Invalid inverse parity denominator")
    return pair.strike/denom

#Independent execution sides
### assert synthetic_sell_forward <= forward_mid <= synthetic_buy_forward
def synthetic_buy_forward(pair: OptionPair) -> float | None:
    if pair.call.ask_coin is None or pair.put.bid_coin is None:
        return None
    denom = (
        1.0 - pair.call.ask_coin + pair.put.bid_coin
    )
    if denom <= 0.0:
        raise ValueError("Invalid synthetic-buy denominator")
    return pair.strike/denom

def synthetic_sell_forward(pair: OptionPair) -> float | None:
    if pair.call.bid_coin is None or pair.put.ask_coin is None:
        return None
    denom = (
        1.0 - pair.call.bid_coin + pair.put.ask_coin
    )
    if denom <= 0.0:
        raise ValueError("Invalid synthetic-sell denominator")
    return pair.strike/denom

#Expiry aggregation
def aggregate_expiry_forward(points: list[ParityPoint]) -> ExpiryForward:
    """ Generates aggregated implied forward"""
    ...

def compare_with_future(
        expiry_forward: ExpiryForward,
        future: FutureQuote | None,
) -> BasisComparison:
    """Compares aggregated implied forward against the traded future."""
    ...