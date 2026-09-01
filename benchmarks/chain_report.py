from deribit.forward_curve import ForwardCurveResult

def format_accounting_report(result: ForwardCurveResult) -> str:
    lines = []

    lines.append("=== FORWARD CURVE DATA ===")
    lines.append(f"Raw option rows:             {result.raw_option_count}")
    lines.append(f"Normalized option rows:      {len(result.quotes)}")
    lines.append(f"Normalization drops:         {len(result.chain_issues)}")
    lines.append(f"Complete call-put pairs:     {len(result.evaluated_pairs)}")
    lines.append(f"Pairing-issue rows:          {result.pairing_issue_quote_count}")
    lines.append(f"Diagnostic midpoint eligible: {result.diagnostic_pair_count}")
    lines.append(f"Synthetic-buy eligible:       {result.synthetic_buy_pair_count}")
    lines.append(f"Synthetic-sell eligible:      {result.synthetic_sell_count if hasattr(result, 'synthetic_sell_count') else result.synthetic_sell_pair_count}")
    lines.append(f"Expiry forwards produced:    {len(result.expiry_forwards)}")
    lines.append(f"Expiry groups not aggregated:{len(result.expiry_issues)}")
    lines.append("\nNote: Rule counts are attribution counts and need not sum to the number of unique excluded pairs.\n")

    raw_denom = result.raw_option_count or 1
    chain_counts = {}
    for issue in result.chain_issues:
        chain_counts[issue.reason] = chain_counts.get(issue.reason, 0) + 1

    lines.append("--- 1. NORMALIZATION ATTRITION (Denominator: Raw Option Rows) ---")
    lines.append(f"{'Reason':<30} {'Rows':<8} {'Fraction':<10}")
    lines.append("-" * 50)
    for reason, count in sorted(chain_counts.items()):
        lines.append(f"{reason:<30} {count:<8} {count / raw_denom:.2%}")
    if not chain_counts:
        lines.append("No normalization drops recorded.")

    norm_denom = len(result.quotes) or 1
    pairing_counts = {}
    for issue in result.pairing_issues:
        pairing_counts[issue.reason] = pairing_counts.get(issue.reason, 0) + len(issue.source_row_ids)

    lines.append("\n--- 2. PAIRING ATTRITION (Denominator: Normalized Option Rows) ---")
    lines.append(f"{'Reason':<30} {'Rows':<8} {'Fraction':<10}")
    lines.append("-" * 50)
    for reason, count in sorted(pairing_counts.items()):
        lines.append(f"{reason:<30} {count:<8} {count / norm_denom:.2%}")
    if not pairing_counts:
        lines.append("No pairing drops recorded.")

    pair_denom = len(result.evaluated_pairs) or 1
    lines.append("\n--- 3. QUOTE HYGIENE ATTRITION (Denominator: Complete Call-Put Pairs) ---")
    lines.append(f"{'Hygiene Reason':<30} {'Pairs':<8} {'Fraction':<10} {'Impact':<25}")
    lines.append("-" * 75)
    for fc in result.filter_counts:
        impact = "Midpoint and one side" if "bid" in fc.reason else ("All uses" if "crossed" in fc.reason else "Midpoint")
        lines.append(f"{fc.reason:<30} {fc.pair_count:<8} {fc.fraction:.2%}      {impact:<25}")

    return "\n".join(lines)

def format_expiry_comparison_table(result: ForwardCurveResult) -> str:
    lines = [
        "\n=== EXPIRY FORWARD VS FUTURE COMPARISON TABLE ===",
        f"{'Expiry':<16} {'Pairs':<5} {'Opt Fwd':<10} {'Fut Bid':<10} {'Fut Mid':<10} "
        f"{'Fut Mark':<10} {'Fut Ask':<10} {'Bps(Mid)':<9} {'Bps(Mark)':<10} "
        f"{'MAD':<7} {'Status':<18}",
        "-" * 115,
    ]

    for comp in result.comparisons:
        ef = next(
            f for f in result.expiry_forwards
            if (f.underlying_index, f.expiration_timestamp) == (comp.underlying_index, comp.expiration_timestamp)
        )

        fut_mid = (
            0.5 * (comp.future_bid + comp.future_ask)
            if comp.future_bid is not None and comp.future_ask is not None and comp.future_bid <= comp.future_ask
            else None
        )

        basis_mid_bps = (
            10_000.0 * (comp.implied_forward / fut_mid - 1.0)
            if fut_mid is not None
            else None
        )

        mid_str = f"{fut_mid:,.1f}" if fut_mid is not None else "N/A"
        mark_str = f"{comp.future_mark:,.1f}" if comp.future_mark is not None else "N/A"
        bid_str = f"{comp.future_bid:,.1f}" if comp.future_bid is not None else "N/A"
        ask_str = f"{comp.future_ask:,.1f}" if comp.future_ask is not None else "N/A"
        bps_mid_str = f"{basis_mid_bps:+.1f}" if basis_mid_bps is not None else "N/A"
        bps_mark_str = f"{comp.basis_bps:+.1f}" if comp.basis_bps is not None else "N/A"

        lines.append(
            f"{comp.underlying_index:<16} {ef.pair_count:<5} {comp.implied_forward:<10,.1f} "
            f"{bid_str:<10} {mid_str:<10} {mark_str:<10} {ask_str:<10} "
            f"{bps_mid_str:<9} {bps_mark_str:<10} {ef.dispersion_mad:<7.1f} {comp.status:<18}"
        )

    return "\n".join(lines)