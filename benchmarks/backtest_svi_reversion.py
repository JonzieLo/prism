import argparse
import math
import numpy as np
from datetime import datetime, timezone
from deribit.store import SnapshotStore, fetch_and_save_snapshot
from deribit.strategy.svi_reversion import SVIReversionStrategy


def main():
    parser = argparse.ArgumentParser(description="Run SVI Dislocation Strategy Backtest.")
    parser.add_argument("--db", default="snapshots.db", help="SQLite database file")
    parser.add_argument("--currency", default="BTC", help="Base currency (BTC/ETH)")
    parser.add_argument("--z-thresh", type=float, default=1.0, help="Entry threshold in multiples of half-spread")
    parser.add_argument("--limit", type=int, default=100, help="Max historical snapshot lookback limit")
    parser.add_argument("--taker-spreads", action="store_true", help="Cross bid-ask spreads (default: midpoints)")
    args = parser.parse_args()

    strategy = SVIReversionStrategy(
        db_path=args.db,
        currency=args.currency,
        z_score_threshold=args.z_thresh,
        max_abs_k=0.50,
        use_midpoint=not args.taker_spreads,
    )

    trades = strategy.run_backtest(snapshot_limit=args.limit)
    print("=== SVI REVERSION STRATEGY BACKTEST RESULTS (EXACT DERIBIT FEES) ===")
    print(f"Execution Mode:          {'ORDERBOOK SPREADS (Crossing Bid/Ask)' if args.taker_spreads else 'MIDPOINT (Alpha Reversion)'}")
    print(f"Dislocation Hurdle:      |z| >= {args.z_thresh:.1f}x half-spread")
    if not trades:
        print("No dislocation trades executed.")
        return

    gross_pnls = np.array([t.target_pnl_usd + t.atm_hedge_pnl_usd + t.delta_hedge_pnl_usd for t in trades])
    net_pnls = np.array([t.net_pnl_usd for t in trades])
    fees = np.array([t.fee_drag_usd for t in trades])

    total_trades = len(trades)
    gross_wins = np.sum(gross_pnls > 0)
    gross_win_rate = gross_wins / total_trades
    gross_total = np.sum(gross_pnls)
    gross_avg = np.mean(gross_pnls)
    gross_std = np.std(gross_pnls)
    gross_sharpe = (gross_avg / gross_std) * math.sqrt(35040) if gross_std > 0 else 0.0

    net_wins = np.sum(net_pnls > 0)
    net_win_rate = net_wins / total_trades
    net_total = np.sum(net_pnls)
    net_avg = np.mean(net_pnls)
    net_std = np.std(net_pnls)
    net_sharpe = (net_avg / net_std) * math.sqrt(35040) if net_std > 0 else 0.0

    print("\n--- 1. PURE MODEL ALPHA (IF FEES = $0.00) ---")
    print(f"Gross Win Rate:           {gross_win_rate:.2%} ({gross_wins}/{total_trades} trades predicted correctly)")
    print(f"Cumulative Gross Alpha:   ${gross_total:,.2f}")
    print(f"Average Gross Edge/Trade: ${gross_avg:+,.2f}")
    print(f"Gross Alpha Sharpe:       {gross_sharpe:+.2f}")

    print("\n--- 2. REALIZED EXECUTION (AFTER EXACT DERIBIT FEES) ---")
    print(f"Net Win Rate:             {net_win_rate:.2%}")
    print(f"Cumulative Net P&L:       ${net_total:,.2f}")
    print(f"Average Net P&L/Trade:    ${net_avg:+,.2f}")
    print(f"Average Fee Drag/Trade:   ${np.mean(fees):,.2f}")
    print(f"Net Realized Sharpe:      {net_sharpe:+.2f}")

    meta_cache: dict[int, str] = {}
    store = SnapshotStore(args.db)
    def get_time_str(snap_id: int) -> str:
        if snap_id not in meta_cache:
            meta = store.load_snapshot_metadata(snap_id)
            if meta and meta.timestamp_ns:
                dt = datetime.fromtimestamp(meta.timestamp_ns / 1e9, tz=timezone.utc)
                meta_cache[snap_id] = dt.strftime("%m-%d %H:%M")
            else:
                meta_cache[snap_id] = f"#{snap_id}"
        return meta_cache[snap_id]
    sorted_trades = sorted(
        trades,
        key=lambda t: (t.target_pnl_usd + t.atm_hedge_pnl_usd + t.delta_hedge_pnl_usd),
        reverse=True,
    )

    print("\n--- TOP 10 TRADES (RANKED BY GROSS ALPHA) ---")
    print(f"{'Snapshots':<12} {'Time (UTC)':<12} {'Instrument':<22} {'Side':<16} {'Z-Score':<9} {'Gross PnL':<12} {'Fee Drag':<10} {'Net PnL':<10}")
    print("-" * 80)
    for t in sorted_trades[:10]:
        gross = t.target_pnl_usd + t.atm_hedge_pnl_usd + t.delta_hedge_pnl_usd
        snap_range = f"#{t.entry_snapshot_id}->#{t.exit_snapshot_id}"
        time_str = get_time_str(t.entry_snapshot_id)
        print(
            f"{snap_range:<12} {time_str:<12} {t.target_instrument:<22} {t.side:<16} "
            f"{t.residual_z_score:<9.2f} ${gross:<+11.2f} ${t.fee_drag_usd:<9.2f} ${t.net_pnl_usd:<+9.2f}"
        )

if __name__ == "__main__":
    main()