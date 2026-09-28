import argparse
import math
from deribit.strategy.svi_reversion import SVIReversionStrategy


def main():
    parser = argparse.ArgumentParser(description="Run SVI Dislocation Strategy Backtest.")
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--threshold-bps", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-loo", action="store_true", help="Disable Leave-One-Out fitting")
    args = parser.parse_args()

    strategy = SVIReversionStrategy(
        db_path=args.db,
        currency=args.currency,
        entry_threshold_bps=args.threshold_bps,
        use_leave_one_out=not args.no_loo,
    )

    trades = strategy.run_backtest(snapshot_limit=args.limit)

    if not trades:
        print("No dislocation trades executed.")
        return

    net_pnls = [t.net_pnl_usd for t in trades]
    total_trades = len(trades)
    winning_trades = sum(1 for pnl in net_pnls if pnl > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    total_pnl = sum(net_pnls)
    avg_pnl = total_pnl / total_trades
    
    variance = sum((x - avg_pnl) ** 2 for x in net_pnls) / total_trades if total_trades > 1 else 0.0
    std_pnl = math.sqrt(variance)
    
    sharpe = (avg_pnl / std_pnl) * math.sqrt(365 * 24 * 4) if std_pnl > 0 else 0.0

    print("\n" + "="*60)
    print("=== SVI REVERSION STRATEGY BACKTEST RESULTS (MILESTONE M8) ===")
    print("="*60)
    print(f"Leave-One-Out (LOO) Fit:  {strategy.use_leave_one_out}")
    print(f"Total Executed Trades:    {total_trades}")
    print(f"Win Rate:                 {win_rate:.2%}")
    print(f"Cumulative Net PnL:       ${total_pnl:,.2f}")
    print(f"Average PnL per Trade:    ${avg_pnl:,.2f}")
    print(f"PnL Std Deviation:        ${std_pnl:,.2f}")
    print(f"Backtested Sharpe Ratio:  {sharpe:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()