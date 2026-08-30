from collections import Counter,defaultdict
from dataclasses import dataclass

from .chain import *
from .forwards import *
from .hygiene import *

@dataclass(frozen=True)
class FilterCount:
    reason: str
    pair_count: int
    fraction: float

@dataclass(frozen=True)
class ExpiryIssue:
    expiration_timestamp: int
    underlying_index:str
    reason: str

@dataclass(frozen=True)
class ForwardCurveResult:
    raw_option_count: int
    quotes: tuple[OptionQuote, ...]
    chain_issues: tuple[ChainIssue,...]
    paired_quote_count: int
    pairing_issue_quote_count: int
    pairing_issues: tuple[PairingIssue, ...]
    evaluated_pairs: tuple[EvaluatedPair, ...]
    diagnostic_pair_count: int
    synthetic_buy_pair_count: int
    synthetic_sell_pair_count: int
    filter_counts: tuple[FilterCount, ...]
    expiry_forwards: tuple[ExpiryForward, ...]
    comparisons: tuple[BasisComparison, ...]
    expiry_issues: tuple[ExpiryIssue, ...]

def _group_points(points: list[ParityPoint]) -> dict[tuple[str, int], list[ParityPoint]]:
    return dict()

def _filter_counts(evaluated: list[EvaluatedPair]) -> tuple[FilterCount, ...]:
    total = len(evaluated)
    affected: Counter[IssueCode] = Counter()
    for item in evaluated:
        for code in {issue.code for issue in item.issues}:
            affected[code] += 1
    return tuple(
        FilterCount(
            reason=code.value,
            pair_count=count,
            fraction=count/total if total else 0.0,
        )
        for code, count in sorted(affected.item(), key=lambda item: item[0].value)
    )

def build_forward_curve(
        snapshot: dict
) -> ForwardCurveResult:
    raw_option_count = len(snapshot.get("options", {}).get("payload") or [])
    quotes, chain_issues = option_chain_report_from_snapshot(snapshot)
    pairs, pairing_issues = pair_calls_and_puts(quotes)
    diagnostic, buys, sells, evaluated = evaluate_and_partition_pairs(pairs)

    diagnostic_by_expiry = _group_points(diagnostic)
    buys_by_expiry = _group_points(buys)
    sells_by_expiry = _group_points(sells)
    paired_keys = {
        (item.pair.underlying_index, item.pair.expiration_timestamp)
        for item in evaluated
    }
    pairing_issue_keys = {
        (underlying_index, expiration)
        for underlying_index, expiration, _ in (
            issue.key for issue in pairing_issues
        ) if underlying_index is not None
    }
    all_keys = paired_keys | pairing_issue_keys
    futures = futures_from_snapshot(snapshot)

    expiry_forwards: list[ExpiryForward] = []
    comparisons: list[BasisComparison] = []
    expiry_issues: list[ExpiryIssue] = []
    for underlying_index, expiration in sorted(all_keys, key=lambda key:key[1]):
        key = (underlying_index, expiration)
        points = diagnostic_by_expiry.get(key,[])
        if not points:
            reason = (
                "no_diagnostic_eligible_pairs"
                if key in paired_keys
                else "no_complete_call_put_pairs"
            )
            expiry_issues.append(
                ExpiryIssue(
                expiration_timestamp=expiration,
                underlying_index=underlying_index,
                reason=reason,
                )
            )
            continue

        expiry_forward = aggregate_expiry_forward(
            points,
            buys_by_expiry.get(key, []),
            sells_by_expiry.get(key, []),
        )
        expiry_forwards.append(expiry_forward)

        future = futures.get(underlying_index)
        if future is None and underlying_index.startswith("SYN."): #synthetic option
            future = futures.get(underlying_index.removeprefix("SYN."))
        comparisons.append(
            compare_with_future(
            expiry_forward, future
            )
        )

    paired_ids = {
        source_id for pair in pairs
        for source_id in (pair.call.source_row_id, pair.put.source_row_id)
    }
    pairing_issue_ids = {
        source_id for issue in pairing_issues
        for source_id in issue.source_row_ids
    }
    normalized_ids = {quote.source_row_id for quote in quotes}
    
    if paired_ids & pairing_issue_ids:
        raise ValueError("Paired and pairing-issue rows overlap")
    if normalized_ids != paired_ids | pairing_issue_ids:
        raise ValueError("Normalized option rows do not reconcile")
    if raw_option_count != len(quotes) + len(chain_issues):
        raise ValueError("Raw option rows do not reconcile")
    
    return ForwardCurveResult(
        raw_option_count=raw_option_count,
        quotes=tuple(quotes),
        chain_issues=tuple(chain_issues),
        paired_quote_count=len(paired_ids),
        pairing_issue_quote_count=len(pairing_issue_ids),
        pairing_issues=tuple(pairing_issues),
        evaluated_pairs=tuple(evaluated),
        diagnostic_pair_count=len(diagnostic),
        synthetic_buy_pair_count=len(buys),
        synthetic_sell_pair_count=len(sells),
        filter_counts=_filter_counts(evaluated),
        expiry_forwards=tuple(expiry_forwards),
        comparisons=tuple(comparisons),
        expiry_issues=tuple(expiry_issues),
    )