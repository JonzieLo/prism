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
from deribit.pricing.fees import calculate_option_combo_fee, calculate_future_fee
from deribit.store import SnapshotStore


def run_basis_arbitrage_scan(
    db_path: str = "snapshots.db",
    currency: str = "BTC",
    min_edge_bps: float = 0.0,
    max_server_skew_ms: float = 50.0,
    max_strike_distance_pct: float = 0.20,  # <20% of spot
):
    store = SnapshotStore(db_path)
    snapshot_ids = [s.snapshot_id for s in store.list_snapshots(currency=currency, limit=1000)]
    snapshot_ids.sort()

    if not snapshot_ids:
        print("No snapshots found in database.")
        return

    print("=" * 80)
    print("=== DERIBIT SYNTHETIC FORWARD BASIS ARBITRAGE SCANNER ===")
    print(f"Target Min Net Edge:   {min_edge_bps:.1f} bps")
    print(f"Max Server Skew Filter:{max_server_skew_ms:.1f} ms")
    print("=" * 80)

    reversals = []
    conversions = []
    evaluated_pairs_count = 0
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

        index_px = snap_data.get("index", {}).get("payload", {}).get("index_price", 0.0)
        if index_px <= 0.0:
            continue

        futures_map = {comp.underlying_index: comp for comp in result.comparisons}

        for eval_pair in result.evaluated_pairs:
            pair = eval_pair.pair
            comp = futures_map.get(pair.underlying_index)
            if not comp or comp.future_bid is None or comp.future_ask is None:
                continue

            evaluated_pairs_count += 1
            fwd = comp.implied_forward
            strike = pair.strike

            if abs(math.log(strike / fwd)) > max_strike_distance_pct:
                continue

            # Check Reversal: Buy synthetic forward, sell future
            call_ask_coin = pair.call.ask_coin
            put_bid_coin = pair.put.bid_coin

            if (
                eval_pair.synthetic_buy_eligible
                and call_ask_coin is not None
                and put_bid_coin is not None
                and call_ask_coin > 0.0
                and put_bid_coin > 0.0
            ):
                denom = 1.0 - call_ask_coin + put_bid_coin
                if denom > 0.0:
                    synth_buy_fwd = strike / denom
                    fut_bid = comp.future_bid

                    gross_edge_usd = fut_bid - synth_buy_fwd
                    call_prem_usd = call_ask_coin * index_px
                    put_prem_usd = put_bid_coin * index_px

                    opt_combo_fee_usd = calculate_option_combo_fee(
                        buy_legs=[(call_prem_usd, 1.0)],   # Bought Call
                        sell_legs=[(put_prem_usd, 1.0)],  # Sold Put
                        index_price=index_px,
                        is_taker=True,
                    )

                    fut_fee_usd = calculate_future_fee(fut_bid, contracts=1.0, is_taker=False)

                    net_edge_usd = gross_edge_usd - (opt_combo_fee_usd + fut_fee_usd)
                    net_edge_bps = (net_edge_usd / fut_bid) * 10_000.0

                    if min_edge_bps <= net_edge_bps < 500.0:
                        reversals.append({
                            "snapshot_id": snap_id,
                            "skew_ms": meta.server_skew_ms if meta else 0.0,
                            "underlying": pair.underlying_index,
                            "strike": strike,
                            "direction": "REVERSAL",
                            "synthetic_price": synth_buy_fwd,
                            "future_price": fut_bid,
                            "call_prem": call_prem_usd,
                            "put_prem": put_prem_usd,
                            "fees_usd": opt_combo_fee_usd + fut_fee_usd,
                            "net_bps": net_edge_bps,
                            "locked_usd": net_edge_usd,
                        })

            # Check Conversion: Sell synthetic forward, buy future
            call_bid_coin = pair.call.bid_coin
            put_ask_coin = pair.put.ask_coin

            if (
                eval_pair.synthetic_sell_eligible
                and call_bid_coin is not None
                and put_ask_coin is not None
                and call_bid_coin > 0.0
                and put_ask_coin > 0.0
            ):
                denom = 1.0 - call_bid_coin + put_ask_coin
                if denom > 0.0:
                    synth_sell_fwd = strike / denom
                    fut_ask = comp.future_ask

                    gross_edge_usd = synth_sell_fwd - fut_ask
                    call_prem_usd = call_bid_coin * index_px
                    put_prem_usd = put_ask_coin * index_px

                    opt_combo_fee_usd = calculate_option_combo_fee(
                        buy_legs=[(put_prem_usd, 1.0)],    # Bought Put
                        sell_legs=[(call_prem_usd, 1.0)],  # Sold Call
                        index_price=index_px,
                        is_taker=True,
                    )

                    fut_fee_usd = calculate_future_fee(fut_ask, contracts=1.0, is_taker=True)

                    net_edge_usd = gross_edge_usd - (opt_combo_fee_usd + fut_fee_usd)
                    net_edge_bps = (net_edge_usd / fut_ask) * 10_000.0

                    if min_edge_bps <= net_edge_bps < 500.0:
                        conversions.append({
                            "snapshot_id": snap_id,
                            "skew_ms": meta.server_skew_ms if meta else 0.0,
                            "underlying": pair.underlying_index,
                            "strike": strike,
                            "direction": "CONVERSION",
                            "synthetic_price": synth_sell_fwd,
                            "future_price": fut_ask,
                            "call_prem": call_prem_usd,
                            "put_prem": put_prem_usd,
                            "fees_usd": opt_combo_fee_usd + fut_fee_usd,
                            "net_bps": net_edge_bps,
                            "locked_usd": net_edge_usd,
                        })

    total_opps = len(reversals) + len(conversions)
    print(f"\nEvaluated Snapshots:     {len(snapshot_ids)} (Skipped {skipped_high_skew} with server skew > {max_server_skew_ms}ms)")
    print(f"Evaluated Strike Pairs:  {evaluated_pairs_count}")
    print(f"Executable Reversals:    {len(reversals)}")
    print(f"Executable Conversions:  {len(conversions)}")

    if total_opps > 0:
        all_net = [x["net_bps"] for x in reversals + conversions]
        all_usd = [x["locked_usd"] for x in reversals + conversions]
        print(f"Mean Net Basis Edge:     +{np.mean(all_net):.2f} bps")
        print(f"Max Net Basis Edge:      +{np.max(all_net):.2f} bps")
        print(f"Cumulative Locked P&L:   ${np.sum(all_usd):,.2f}")
        print("\n--- IDENTIFIED TRUE CROSSES ---")
        print(f"{'Snapshot':<10} {'Underlying':<16} {'Strike':<10} {'Type':<12} {'Fees':<9} {'Net Edge':<12} {'Locked USD':<10}")
        print("-" * 80)
        for opp in sorted(reversals + conversions, key=lambda x: x["net_bps"], reverse=True)[:10]:
            print(f"#{opp['snapshot_id']:<9} {opp['underlying']:<16} ${opp['strike']:<9,.0f} {opp['direction']:<12} ${opp['fees_usd']:<8.2f} +{opp['net_bps']:<10.1f}bps ${opp['locked_usd']:,.2f}")
    else:
        print("\nNo crossings exceeded the exact fee hurdle within the ±20% strike window. Markets were arbitrage-free.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--min-edge-bps", type=float, default=0.0)
    parser.add_argument("--max-skew", type=float, default=50.0)
    parser.add_argument("--max-strike-pct", type=float, default=0.20)
    args = parser.parse_args()

    run_basis_arbitrage_scan(args.db, args.currency, args.min_edge_bps, args.max_skew, args.max_strike_pct)