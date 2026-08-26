import math
from dataclasses import dataclass
from typing import Any


MILLISECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0 * 1000.0


@dataclass(frozen=True)
class OptionQuote:
    instrument_name: str
    option_type: str
    strike: float
    expiration_timestamp: int
    index_price: float
    forward: float
    rate: float
    tau: float
    bid_coin: float | None
    ask_coin: float | None
    mark_coin: float | None
    deribit_mark_iv: float | None

    @property
    def mid_coin(self) -> float | None:
        if (
            self.bid_coin is None
            or self.ask_coin is None
            or self.bid_coin <= 0.0
            or self.ask_coin < self.bid_coin
        ):
            return None
        return 0.5 * (self.bid_coin + self.ask_coin)

    @property
    def mid_usd(self) -> float | None:
        mid = self.mid_coin
        return None if mid is None else self.index_price * mid


def _snapshot_time_ms(snapshot: dict[str, Any]) -> float:
    received = [
        response["received_at_ns"]
        for response in snapshot.values()
        if response.get("received_at_ns") is not None
    ]
    if not received:
        raise ValueError("snapshot has no received_at_ns timestamps")
    return max(received) / 1_000_000.0


def option_chain_from_snapshot(snapshot: dict[str, Any],) -> list[OptionQuote]:
    instruments = snapshot.get("instruments", {}).get("payload") or []
    market_rows = snapshot.get("options", {}).get("payload") or []
    index_payload = snapshot.get("index", {}).get("payload") or {}
    index_price = index_payload.get("index_price")
    if index_price is None or not math.isfinite(index_price) or index_price <= 0:
        raise ValueError("snapshot contains no positive index price")

    snapshot_ms = _snapshot_time_ms(snapshot)
    specs = {
        row["instrument_name"]: row
        for row in instruments
        if row.get("kind") == "option"
    }
    chain: list[OptionQuote] = []

    for market in market_rows:
        spec = specs.get(market.get("instrument_name"))
        if spec is None:
            continue

        forward = market.get("underlying_price")
        expiration = spec.get("expiration_timestamp")
        strike = spec.get("strike")
        if forward is None or expiration is None or strike is None:
            continue
        if forward <= 0.0 or strike <= 0.0:
            continue

        tau = (expiration - snapshot_ms) / MILLISECONDS_PER_YEAR
        if tau <= 0.0:
            continue

        rate = math.log(forward / index_price) / tau
        mark_iv_percent = market.get("mark_iv")

        chain.append(
            OptionQuote(
                instrument_name=spec["instrument_name"],
                option_type=spec["option_type"],
                strike=float(strike),
                expiration_timestamp=int(expiration),
                index_price=float(index_price),
                forward=float(forward),
                rate=rate,
                tau=tau,
                bid_coin=market.get("bid_price"),
                ask_coin=market.get("ask_price"),
                mark_coin=market.get("mark_price"),
                deribit_mark_iv=(
                    mark_iv_percent / 100.0
                    if mark_iv_percent is not None
                    else None
                ),
            )
        )

    return chain


def select_expiry(
    chain: list[OptionQuote],
    option_type: str,
    expiration_timestamp: int | None = None,
) -> list[OptionQuote]:
    candidates = [
        quote
        for quote in chain
        if quote.option_type == option_type and quote.mid_coin is not None
    ]
    if expiration_timestamp is None:
        counts: dict[int, int] = {}
        for quote in candidates:
            counts[quote.expiration_timestamp] = (
                counts.get(quote.expiration_timestamp, 0) + 1
            )
        if not counts:
            raise ValueError(f"no two-sided {option_type} quotes in snapshot")
        expiration_timestamp = max(counts, key=counts.get)

    selected = [
        quote
        for quote in candidates
        if quote.expiration_timestamp == expiration_timestamp
    ]
    if not selected:
        raise ValueError(
            f"no two-sided {option_type} quotes for expiry "
            f"{expiration_timestamp}"
        )
    return sorted(selected, key=lambda quote: quote.strike)
