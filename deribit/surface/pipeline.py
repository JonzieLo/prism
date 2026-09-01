import math
from collections import defaultdict

from deribit.chain import OptionQuote
from deribit.forwards import ExpiryForward
from deribit.pricing import Black76Model
from .context import SurfaceExpiryContext
from .filter_policy import SurfaceFilterPolicy
from .observations import *


def _relative_spread(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0.0 or ask < bid:
        return None
    mid = 0.5 * (bid + ask)
    return (ask - bid) / mid


def build_surface_observations(
    snapshot_id: int,
    quotes: list[OptionQuote],
    expiry_forwards: list[ExpiryForward],
    policy: SurfaceFilterPolicy = SurfaceFilterPolicy(),
) -> SurfaceObservationResult:
    black76 = Black76Model()
    observations: list[SurfaceObservation] = []
    exclusions: list[SurfaceExclusion] = []

    context_map = {}
    for ef in expiry_forwards:
        key = (ef.underlying_index, ef.expiration_timestamp)
        sample = next((q for q in quotes if (q.underlying_index, q.expiration_timestamp) == key), None)
        if sample is None:
            continue
        rate = math.log(ef.implied_forward / sample.index_price) / sample.tau
        context_map[key] = SurfaceExpiryContext(
            underlying_index=ef.underlying_index,
            expiration_timestamp=ef.expiration_timestamp,
            index_price=sample.index_price,
            forward=ef.implied_forward,
            tau=sample.tau,
            rate=rate,
        )

    grouped = defaultdict(list)
    for q in quotes:
        grouped[(q.underlying_index, q.expiration_timestamp, q.strike)].append(q)

    for (underlying_index, expiry_ts, strike), group in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][2])):
        key = (underlying_index, expiry_ts)
        context = context_map.get(key)

        if context is None:
            exclusions.append(
                SurfaceExclusion(
                    source_row_id=group[0].source_row_id,
                    instrument_name=group[0].instrument_name,
                    underlying_index=underlying_index,
                    expiration_timestamp=expiry_ts,
                    strike=strike,
                    reason=SurfaceExclusionCode.MISSING_EXPIRY_CONTEXT,
                    detail="No derived forward available for expiry",
                )
            )
            continue

        if strike < context.forward:
            desired_type = "put"
        elif strike > context.forward:
            desired_type = "call"
        else:
            valid_calls = [q for q in group if q.option_type == "call" and q.mid_coin is not None]
            valid_puts = [q for q in group if q.option_type == "put" and q.mid_coin is not None]
            c_sprd = _relative_spread(valid_calls[0].bid_coin, valid_calls[0].ask_coin) if valid_calls else None
            p_sprd = _relative_spread(valid_puts[0].bid_coin, valid_puts[0].ask_coin) if valid_puts else None
            
            if c_sprd is not None and (p_sprd is None or c_sprd <= p_sprd):
                desired_type = "call"
            elif p_sprd is not None:
                desired_type = "put"
            else:
                desired_type = "call"

        candidates = [q for q in group if q.option_type == desired_type]

        if not candidates:
            exclusions.append(
                SurfaceExclusion(
                    source_row_id=group[0].source_row_id,
                    instrument_name=group[0].instrument_name,
                    underlying_index=underlying_index,
                    expiration_timestamp=expiry_ts,
                    strike=strike,
                    reason=SurfaceExclusionCode.MISSING_CANONICAL_OTM_LEG,
                    detail=f"Missing canonical OTM {desired_type} leg",
                )
            )
            continue

        if len(candidates) > 1:
            exclusions.append(
                SurfaceExclusion(
                    source_row_id=candidates[0].source_row_id,
                    instrument_name=candidates[0].instrument_name,
                    underlying_index=underlying_index,
                    expiration_timestamp=expiry_ts,
                    strike=strike,
                    reason=SurfaceExclusionCode.DUPLICATE_OPTION_LEG,
                    detail=f"Multiple {desired_type} quotes found for strike",
                )
            )
            continue

        quote = candidates[0]

        # Bid/Ask Hygiene
        if quote.bid_coin is None or quote.ask_coin is None or quote.bid_coin <= 0.0:
            exclusions.append(
                SurfaceExclusion(
                    source_row_id=quote.source_row_id,
                    instrument_name=quote.instrument_name,
                    underlying_index=underlying_index,
                    expiration_timestamp=expiry_ts,
                    strike=strike,
                    reason=SurfaceExclusionCode.INVALID_BID_ASK,
                    detail="Missing or zero bid price",
                )
            )
            continue

        if quote.ask_coin < quote.bid_coin:
            exclusions.append(
                SurfaceExclusion(
                    source_row_id=quote.source_row_id,
                    instrument_name=quote.instrument_name,
                    underlying_index=underlying_index,
                    expiration_timestamp=expiry_ts,
                    strike=strike,
                    reason=SurfaceExclusionCode.CROSSED_BOOK,
                    detail="Crossed bid-ask spread",
                )
            )
            continue

        rel_spread = (quote.ask_coin - quote.bid_coin) / quote.mid_coin
        if policy.max_relative_spread is not None and rel_spread > policy.max_relative_spread:
            exclusions.append(
                SurfaceExclusion(
                    source_row_id=quote.source_row_id,
                    instrument_name=quote.instrument_name,
                    underlying_index=underlying_index,
                    expiration_timestamp=expiry_ts,
                    strike=strike,
                    reason=SurfaceExclusionCode.SPREAD_POLICY,
                    detail=f"Relative spread {rel_spread:.2%} exceeds policy max {policy.max_relative_spread:.2%}",
                )
            )
            continue

        # Invert Mid IV
        try:
            mid_iv = black76.implied_vol(
                quote.mid_usd,
                context.forward,
                quote.strike,
                context.tau,
                context.rate,
                quote.option_type,
            )
        except Exception as exc:
            exclusions.append(
                SurfaceExclusion(
                    source_row_id=quote.source_row_id,
                    instrument_name=quote.instrument_name,
                    underlying_index=underlying_index,
                    expiration_timestamp=expiry_ts,
                    strike=strike,
                    reason=SurfaceExclusionCode.MID_IV_FAILURE,
                    detail=str(exc),
                )
            )
            continue

        # Optional Bid / Ask IVs
        bid_iv = None
        ask_iv = None
        try:
            bid_iv = black76.implied_vol(
                context.index_price * quote.bid_coin,
                context.forward, quote.strike, context.tau, context.rate, quote.option_type
            )
        except Exception:
            pass

        try:
            ask_iv = black76.implied_vol(
                context.index_price * quote.ask_coin,
                context.forward, quote.strike, context.tau, context.rate, quote.option_type
            )
        except Exception:
            pass

        log_k = math.log(quote.strike / context.forward)
        tot_var = mid_iv * mid_iv * context.tau

        observations.append(
            SurfaceObservation(
                source_row_id=quote.source_row_id,
                instrument_name=quote.instrument_name,
                underlying_index=underlying_index,
                expiration_timestamp=expiry_ts,
                index_price=context.index_price,
                forward=context.forward,
                strike=quote.strike,
                tau=context.tau,
                rate=context.rate,
                log_moneyness=log_k,
                option_type=quote.option_type,
                bid_coin=quote.bid_coin,
                ask_coin=quote.ask_coin,
                mid_coin=quote.mid_coin,
                bid_usd=context.index_price * quote.bid_coin,
                ask_usd=context.index_price * quote.ask_coin,
                mid_usd=quote.mid_usd,
                bid_iv=bid_iv,
                mid_iv=mid_iv,
                ask_iv=ask_iv,
                total_variance=tot_var,
                relative_spread=rel_spread,
                open_interest=quote.open_interest,
                volume=quote.volume,
            )
        )

    return SurfaceObservationResult(
        observations=tuple(observations),
        exclusions=tuple(exclusions),
    )