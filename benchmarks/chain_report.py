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