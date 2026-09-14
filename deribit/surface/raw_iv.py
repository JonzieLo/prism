import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from deribit.chain import OptionQuote
from deribit.forwards import ExpiryForward
from deribit.pricing import (
    BachelierModel,
    BinomialModel,
    Black76Model,
    BlackScholesModel,
    OptionModel,
)


SurfaceModelName = Literal[
    "black_scholes",
    "black76",
    "bachelier",
    "binomial",
    "inverse",
]

SURFACE_MODEL_NAMES: tuple[SurfaceModelName, ...] = (
    "black_scholes",
    "black76",
    "bachelier",
    "binomial",
    "inverse",
)


@dataclass(frozen=True)
class RawIVPoint:
    snapshot_id: int
    model_name: SurfaceModelName
    instrument_name: str
    underlying_index: str
    expiration_timestamp: int
    option_type: str
    index_price: float
    forward: float
    strike: float
    tau: float
    rate: float
    implied_vol: float
    vol_unit: str


@dataclass(frozen=True)
class RawIVDrop:
    snapshot_id: int
    underlying_index: str
    expiration_timestamp: int
    strike: float
    reason: str


@dataclass(frozen=True)
class RawIVResult:
    snapshot_id: int
    model_name: SurfaceModelName
    points: tuple[RawIVPoint, ...]
    drops: tuple[RawIVDrop, ...]


def _model(
    model_name: SurfaceModelName,
    binomial_steps: int,
) -> OptionModel:
    if model_name == "black_scholes":
        return BlackScholesModel()
    if model_name in {"black76", "inverse"}:
        return Black76Model()
    if model_name == "bachelier":
        return BachelierModel()
    if model_name == "binomial":
        return BinomialModel(steps=binomial_steps)
    raise ValueError(f"Unsupported surface model: {model_name}")


def _relative_spread(quote: OptionQuote) -> float:
    if quote.mid_coin is None:
        return math.inf
    return (quote.ask_coin - quote.bid_coin) / quote.mid_coin


def _canonical_quote(
    group: list[OptionQuote],
    forward: float,
) -> OptionQuote | None:
    strike = group[0].strike
    desired_type = "put" if strike < forward else "call"

    if strike == forward:
        candidates = [
            quote for quote in group
            if quote.option_type in {"call", "put"}
            and quote.mid_coin is not None
        ]
        return min(candidates, key=_relative_spread) if candidates else None

    candidates = [
        quote for quote in group
        if quote.option_type == desired_type
        and quote.mid_coin is not None
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def build_raw_iv_points(
    snapshot_id: int,
    quotes: list[OptionQuote],
    expiry_forwards: list[ExpiryForward],
    model_name: SurfaceModelName = "black76",
    *,
    binomial_steps: int = 100,
) -> RawIVResult:
    """Recover raw model IVs without fitting or interpolating a surface."""
    if snapshot_id <= 0:
        raise ValueError("snapshot_id must be positive")
    if model_name not in SURFACE_MODEL_NAMES:
        raise ValueError(f"Unsupported surface model: {model_name}")

    model = _model(model_name, binomial_steps)
    forward_map = {
        (item.underlying_index, item.expiration_timestamp): item.implied_forward
        for item in expiry_forwards
    }
    grouped: dict[tuple[str | None, int, float], list[OptionQuote]] = (
        defaultdict(list)
    )
    for quote in quotes:
        grouped[
            (
                quote.underlying_index,
                quote.expiration_timestamp,
                quote.strike,
            )
        ].append(quote)

    points: list[RawIVPoint] = []
    drops: list[RawIVDrop] = []
    for (underlying_index, expiry, strike), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][2]),
    ):
        display_index = underlying_index or "UNKNOWN"
        forward = forward_map.get((underlying_index, expiry))
        if forward is None:
            drops.append(
                RawIVDrop(
                    snapshot_id,
                    display_index,
                    expiry,
                    strike,
                    "missing expiry forward",
                )
            )
            continue

        quote = _canonical_quote(group, forward)
        if quote is None or quote.mid_usd is None:
            drops.append(
                RawIVDrop(
                    snapshot_id,
                    display_index,
                    expiry,
                    strike,
                    "missing unique two-sided canonical OTM quote",
                )
            )
            continue

        rate = math.log(forward / quote.index_price) / quote.tau
        model_underlying = (
            quote.index_price
            if model_name == "black_scholes"
            else forward
        )

        try:
            implied_vol = model.implied_vol(
                quote.mid_usd,
                model_underlying,
                quote.strike,
                quote.tau,
                rate,
                quote.option_type,
            )
        except ValueError as error:
            drops.append(
                RawIVDrop(
                    snapshot_id,
                    display_index,
                    expiry,
                    strike,
                    str(error),
                )
            )
            continue

        points.append(
            RawIVPoint(
                snapshot_id=snapshot_id,
                model_name=model_name,
                instrument_name=quote.instrument_name,
                underlying_index=display_index,
                expiration_timestamp=expiry,
                option_type=quote.option_type,
                index_price=quote.index_price,
                forward=forward,
                strike=strike,
                tau=quote.tau,
                rate=rate,
                implied_vol=implied_vol,
                vol_unit=(
                    "USD/sqrt(year)"
                    if model_name == "bachelier"
                    else "decimal"
                ),
            )
        )

    return RawIVResult(
        snapshot_id=snapshot_id,
        model_name=model_name,
        points=tuple(points),
        drops=tuple(drops),
    )