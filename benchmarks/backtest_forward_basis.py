"""
Put-Call Parity Forward Arbitrage Backtest (Reversals & Conversions).

Scans snapshots.db for executable top-of-book crossings:
1. Reversal: Traded Future Bid > Synthetic Buy Forward
   -> Buy Call, Sell Put at strike K, Sell Traded Future
2. Conversion: Synthetic Sell Forward > Traded Future Ask
   -> Sell Call, Buy Put at strike K, Buy Traded Future

Calculates net basis capture in basis points (bps) after Deribit fees.
"""

import argparse
import math
import numpy as np

from deribit.forward_curve import build_forward_curve
from deribit.forwards import BasisStatus
from deribit.store import SnapshotStore


def run_basis_arbitrage_scan(
    db_path: str = "snapshots.db",
    currency: str = "BTC",
    opt_taker_bps: float = 3.0,
    fut_taker_bps: float = 1.5,
    min_edge_bps: float = 0.0,
    max_server_skew_ms: float = 50.0,
):
    store = SnapshotStore(db_path)
    snapshot_ids = [s.snapshot_id for s in store.list_snapshots(currency=currency, limit=1000)]
    snapshot_ids.sort()

    if not snapshot_ids:
        print("No snapshots found in database.")
        return

    # Total round-trip fee in bps:
    # 2 option legs (each pays opt_taker_bps) + 1 future leg (pays fut_taker_bps)
    total_fee_bps = (2.0 * opt_taker_bps) + fut_taker_bps

    print("=" * 80)
    print("=== DERIBIT SYNTHETIC FORWARD BASIS ARBITRAGE SCANNER ===")
    print(f"Fee Hurdle:            {total_fee_bps:.1f} bps (2 options @ {opt_taker_bps} + 1 future @ {fut_taker_bps})")
    print(f"Target Min Net Edge:   {min_edge_bps:.1f} bps")
    print(f"Max Server Skew Filter:{max_server_skew_ms:.1f} ms")
    print("=" * 80)

    reversals = []
    conversions = []
    evaluated_comparisons = 0
    skipped_high_skew = 0

    for snap_id in snapshot_ids:
        meta = store.load_snapshot_metadata(snap_id)
        if meta and meta.server_skew_ms and meta.server_skew_ms > max_server_skew_ms:
            skipped_high_skew += 1
            continue

        snap_data = store.load_snapshot(snap_id)
        if not snap_data:
            continue

        try:
            result = build_forward_curve(snap_data)
        except Exception:
            continue

        for comp in result.comparisons:
            evaluated_comparisons += 1
            if comp.status != BasisStatus.PRICE_CROSS.value and comp.status != BasisStatus.OK.value:
                continue

            # Check Reversal: Buy synthetic forward, sell future
            if comp.future_bid and comp.best_synthetic_buy:
                gross_edge_usd = comp.future_bid - comp.best_synthetic_buy
                gross_bps = (gross_edge_usd / comp.future_bid) * 10_000.0
                net_bps = gross_bps - total_fee_bps

                # Sanity: must meet min_edge_bps and not be an absurd wing artifact (>500 bps)
                if min_edge_bps <= net_bps < 500.0:
                    reversals.append({
                        "snapshot_id": snap_id,
                        "skew_ms": meta.server_skew_ms if meta else 0.0,
                        "underlying": comp.underlying_index,
                        "direction": "REVERSAL (Buy Synthetic, Sell Future)",
                        "synthetic_price": comp.best_synthetic_buy,
                        "future_price": comp.future_bid,
                        "gross_bps": gross_bps,
                        "net_bps": net_bps,
                        "locked_usd": (net_bps / 10_000.0) * comp.future_bid,
                    })

            # Check Conversion: Sell synthetic forward, buy future
            if comp.future_ask and comp.best_synthetic_sell:
                gross_edge_usd = comp.best_synthetic_sell - comp.future_ask
                gross_bps = (gross_edge_usd / comp.future_ask) * 10_000.0
                net_bps = gross_bps - total_fee_bps

                if min_edge_bps <= net_bps < 500.0:
                    conversions.append({
                        "snapshot_id": snap_id,
                        "skew_ms": meta.server_skew_ms if meta else 0.0,
                        "underlying": comp.underlying_index,
                        "direction": "CONVERSION (Sell Synthetic, Buy Future)",
                        "synthetic_price": comp.best_synthetic_sell,
                        "future_price": comp.future_ask,
                        "gross_bps": gross_bps,
                        "net_bps": net_bps,
                        "locked_usd": (net_bps / 10_000.0) * comp.future_ask,
                    })

    total_opps = len(reversals) + len(conversions)
    print(f"\nEvaluated Snapshots:     {len(snapshot_ids)} (Skipped {skipped_high_skew} with server skew > {max_server_skew_ms}ms)")
    print(f"Evaluated Expiry Pairs:  {evaluated_comparisons}")
    print(f"Executable Reversals:    {len(reversals)}")
    print(f"Executable Conversions:  {len(conversions)}")

    if total_opps > 0:
        all_net = [x["net_bps"] for x in reversals + conversions]
        all_usd = [x["locked_usd"] for x in reversals + conversions]
        print(f"Mean Net Basis Edge:     +{np.mean(all_net):.2f} bps")
        print(f"Max Net Basis Edge:      +{np.max(all_net):.2f} bps")
        print(f"Cumulative Locked P&L:   ${np.sum(all_usd):,.2f}")
        print("\n--- IDENTIFIED TRUE CROSSES ---")
        for opp in sorted(reversals + conversions, key=lambda x: x["net_bps"], reverse=True)[:10]:
            print(f"  Snapshot #{opp['snapshot_id']:<3} (Skew: {opp['skew_ms']:.1f}ms) | {opp['underlying']} | {opp['direction']}")
            print(f"    Synthetic: ${opp['synthetic_price']:,.1f} | Future: ${opp['future_price']:,.1f} | Net Edge: +{opp['net_bps']:.1f} bps (${opp['locked_usd']:,.2f})")
    else:
        print(f"\nNo crossings exceeded the hurdle after skew filtering. Market quotes were tight and arbitrage-free.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--opt-fee", type=float, default=3.0, help="Options fee in bps")
    parser.add_argument("--fut-fee", type=float, default=1.5, help="Futures fee in bps")
    parser.add_argument("--min-edge-bps", type=float, default=0.0, help="Min net edge hurdle")
    parser.add_argument("--max-skew", type=float, default=50.0, help="Max exchange server skew in ms")
    args = parser.parse_args()

    run_basis_arbitrage_scan(
        args.db,
        args.currency,
        args.opt_fee,
        args.fut_fee,
        args.min_edge_bps,
        args.max_skew,
    )