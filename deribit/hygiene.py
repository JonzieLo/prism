from dataclasses import dataclass
from enum import Enum

from .chain import OptionQuote
from .forwards import (OptionPair,ParityPoint,matched_size,inverse_forward_mid, synthetic_buy_forward, synthetic_sell_forward)


class IssueCode(str, Enum):
    MISSING_BID = "missing_bid"
    MISSING_ASK = "missing_ask"
    ZERO_BID = "zero_bid"
    NON_POSITIVE_ASK = "non_positive_ask"
    CROSSED_BOOK = "crossed_book"
    SPREAD_WIDER_THAN_MID = "spread_wider_than_mid"
    INVALID_MID_DENOMINATOR = "invalid_mid_denominator"
    INVALID_BUY_DENOMINATOR = "invalid_buy_denominator"
    INVALID_SELL_DENOMINATOR = "invalid_sell_denominator"
    MISSING_API_FORWARD = "missing_api_forward"
    MISSING_MARK_IV = "missing_mark_iv"


class Use(str, Enum):
    DIAGNOSTIC_MID = "diagnostic_mid"
    SYNTHETIC_BUY = "synthetic_buy"
    SYNTHETIC_SELL = "synthetic_sell"


@dataclass(frozen=True)
class QuoteIssue:
    code: IssueCode
    instrument_name: str
    leg: str
    message: str
    blocks: frozenset[Use]


@dataclass(frozen=True)
class EvaluatedPair:
    pair: OptionPair
    point: ParityPoint
    issues: tuple[QuoteIssue, ...]

    @property
    def diagnostic_eligible(self) -> bool:
        return not any(
            Use.DIAGNOSTIC_MID in issue.blocks
            for issue in self.issues
        )

    @property
    def synthetic_buy_eligible(self) -> bool:
        return not any(
            Use.SYNTHETIC_BUY in issue.blocks
            for issue in self.issues
        )

    @property
    def synthetic_sell_eligible(self) -> bool:
        return not any(
            Use.SYNTHETIC_SELL in issue.blocks
            for issue in self.issues
        )


def evaluate_leg(
    quote: OptionQuote,
    leg: str,
) -> list[QuoteIssue]:
    issues: list[QuoteIssue] = []

    if quote.bid_coin is None:
        issues.append(
            QuoteIssue(
                IssueCode.MISSING_BID,
                quote.instrument_name,
                leg,
                "bid is missing",
                frozenset(
                    {
                        Use.DIAGNOSTIC_MID,
                        Use.SYNTHETIC_SELL
                        if leg == "call"
                        else Use.SYNTHETIC_BUY,
                    }
                ),
            )
        )
    elif quote.bid_coin <= 0.0:
        issues.append(
            QuoteIssue(
                IssueCode.ZERO_BID,
                quote.instrument_name,
                leg,
                "bid is not positive",
                frozenset(
                    {
                        Use.DIAGNOSTIC_MID,
                        Use.SYNTHETIC_SELL
                        if leg == "call"
                        else Use.SYNTHETIC_BUY,
                    }
                ),
            )
        )

    if quote.ask_coin is None:
        issues.append(
            QuoteIssue(
                IssueCode.MISSING_ASK,
                quote.instrument_name,
                leg,
                "ask is missing",
                frozenset(
                    {
                        Use.DIAGNOSTIC_MID,
                        Use.SYNTHETIC_BUY
                        if leg == "call"
                        else Use.SYNTHETIC_SELL,
                    }
                ),
            )
        )
    elif quote.ask_coin <= 0.0:
        issues.append(
            QuoteIssue(
                IssueCode.NON_POSITIVE_ASK,
                quote.instrument_name,
                leg,
                "ask is not positive",
                frozenset(
                    {
                        Use.DIAGNOSTIC_MID,
                        Use.SYNTHETIC_BUY
                        if leg == "call"
                        else Use.SYNTHETIC_SELL,
                    }
                ),
            )
        )

    if (
        quote.bid_coin is not None
        and quote.ask_coin is not None
        and quote.bid_coin > quote.ask_coin
    ):
        issues.append(
            QuoteIssue(
                IssueCode.CROSSED_BOOK,
                quote.instrument_name,
                leg,
                "bid exceeds ask",
                frozenset(Use),
            )
        )

    if (
        quote.bid_coin is not None
        and quote.ask_coin is not None
        and 0.0 < quote.bid_coin <= quote.ask_coin
    ):
        midpoint = 0.5 * (
            quote.bid_coin + quote.ask_coin
        )
        spread = quote.ask_coin - quote.bid_coin
        if spread > midpoint:
            issues.append(
                QuoteIssue(
                    IssueCode.SPREAD_WIDER_THAN_MID,
                    quote.instrument_name,
                    leg,
                    "spread is wider than option midpoint",
                    frozenset({Use.DIAGNOSTIC_MID}),
                )
            )

    return issues


def evaluate_pair(pair: OptionPair) -> EvaluatedPair:
    issues = [
        *evaluate_leg(pair.call, "call"),
        *evaluate_leg(pair.put, "put"),
    ]

    try:
        forward_mid = inverse_forward_mid(pair)
    except ValueError as error:
        forward_mid = None
        issues.append(
            QuoteIssue(
                IssueCode.INVALID_MID_DENOMINATOR,
                pair.call.instrument_name,
                "pair",
                str(error),
                frozenset({Use.DIAGNOSTIC_MID}),
            )
        )

    try:
        buy_forward = synthetic_buy_forward(pair)
    except ValueError as error:
        buy_forward = None
        issues.append(
            QuoteIssue(
                IssueCode.INVALID_BUY_DENOMINATOR,
                pair.call.instrument_name,
                "pair",
                str(error),
                frozenset({Use.SYNTHETIC_BUY}),
            )
        )

    try:
        sell_forward = synthetic_sell_forward(pair)
    except ValueError as error:
        sell_forward = None
        issues.append(
            QuoteIssue(
                IssueCode.INVALID_SELL_DENOMINATOR,
                pair.call.instrument_name,
                "pair",
                str(error),
                frozenset({Use.SYNTHETIC_SELL}),
            )
        )

    point = ParityPoint(
        expiration_timestamp=pair.expiration_timestamp,
        underlying_index=pair.underlying_index,
        strike=pair.strike,
        forward_mid=forward_mid,
        synthetic_buy_forward=buy_forward,
        synthetic_sell_forward=sell_forward,
        synthetic_buy_max_size=matched_size(
            pair.call.ask_amount,
            pair.put.bid_amount,
        ),
        synthetic_sell_max_size=matched_size(
            pair.call.bid_amount,
            pair.put.ask_amount,
        ),
    )

    for leg, quote in (
        ("call", pair.call),
        ("put", pair.put),
    ):
        if quote.api_forward is None:
            issues.append(
                QuoteIssue(
                    IssueCode.MISSING_API_FORWARD,
                    quote.instrument_name,
                    leg,
                    "API forward unavailable for comparison",
                    frozenset(),
                )
            )

        if quote.deribit_mark_iv is None:
            issues.append(
                QuoteIssue(
                    IssueCode.MISSING_MARK_IV,
                    quote.instrument_name,
                    leg,
                    "mark IV unavailable for post-hoc comparison",
                    frozenset(),
                )
            )

    return EvaluatedPair(
        pair=pair,
        point=point,
        issues=tuple(issues),
    )

def evaluate_and_partition_pairs(
    pairs: list[OptionPair],
) -> tuple[list[ParityPoint], list[ParityPoint], list[ParityPoint], list[EvaluatedPair]]:
    
    evaluated = [evaluate_pair(pair) for pair in pairs]

    diagnostic_points = [
        item.point
        for item in evaluated
        if item.diagnostic_eligible
    ]
    buy_points = [
        item.point
        for item in evaluated
        if item.synthetic_buy_eligible
    ]
    sell_points = [
        item.point
        for item in evaluated
        if item.synthetic_sell_eligible
    ]

    return diagnostic_points, buy_points, sell_points, evaluated