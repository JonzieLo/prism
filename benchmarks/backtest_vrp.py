from __future__ import annotations

import argparse
import math
import numpy as np

from deribit.pricing.black_scholes import Black76Model
from deribit.pricing.inverse import from_forward_greeks
from deribit.store import SnapshotStore
from deribit.strategy.garch import GJR_GARCH
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.snapshot_loader import load_snapshot_observations


def run_vrp_backtest(
    db_path: str = "snapshots.db",
    currency: str = "BTC",
    min_vrp_vol_points: float = 5.0,
    taker_fee_bps: float = 3.0,
):
    store = SnapshotStore(db_path)
    snapshots = store.list_snapshots(currency=currency, limit=500)
    snapshots = sorted(snapshots, key=lambda s: s.timestamp_ns)

    if len(snapshots) < 10:
        print("Need more snapshots in the database to run a multi-step backtest.")
        return

    prices = []
    for s in snapshots:
        snap_data = store.load_snapshot(s.snapshot_id)
        if snap_data and "index" in snap_data:
            idx = snap_data["index"].get("payload", {}).get("index_price")
            if idx and math.isfinite(idx):
                prices.append(idx)

    if len(prices) < 30:
        print("Not enough price points to calibrate GARCH. Simulating return baseline...")
        returns = np.random.normal(0.0005, 0.025, 120)
    else:
        returns = np.diff(np.log(prices))

    garch = GJR_GARCH()
    garch_params = garch.fit(returns)
    sig2_path = garch.filter_variance(returns, garch_params)
    current_var = sig2_path[-1]
    current_shock = returns[-1] - garch_params.mu

    model = Black76Model()
    fee_rate = taker_fee_bps / 10_000.0

    trades = []
    print(f"=== RUNNING VRP HARVESTING BACKTEST ({currency}) ===")
    print(f"GARCH Long-Run Vol: {garch_params.unconditional_vol_annualized * 100:.2f}%")
    print(f"Minimum VRP Entry:  {min_vrp_vol_points:+.2f} vol points\n")

    for i in range(len(snapshots) - 1):
        s_early = snapshots[i]
        s_late = snapshots[i + 1]

        try:
            obs_early = load_snapshot_observations(store, s_early.snapshot_id)
            obs_late = load_snapshot_observations(store, s_late.snapshot_id)
        except Exception:
            continue

        for expiry, early_list in obs_early.observations_by_expiry.items():
            if expiry not in obs_late.observations_by_expiry:
                continue

            late_list = obs_late.observations_by_expiry[expiry]
            if len(early_list) < 6 or len(late_list) < 6:
                continue

            try:
                smile_early = calibrate_svi_slice(early_list, require_butterfly_free=True)
            except Exception:
                continue

            tau_days = smile_early.tau * 365.0
            garch_rv = garch.forecast_realized_vol(
                garch_params, current_var, current_shock, horizon_days=tau_days
            )

            atm_w = smile_early.parameters.total_variance(0.0)
            atm_iv = math.sqrt(atm_w / smile_early.tau)

            vrp = (atm_iv - garch_rv) * 100.0  # in vol points

            if vrp >= min_vrp_vol_points:
                atm_call = min(early_list, key=lambda o: abs(o.strike - smile_early.forward))
                late_map = {o.instrument_name: o for o in late_list}

                late_call = late_map.get(atm_call.instrument_name)
                if not late_call:
                    continue

                entry_opt_price = atm_call.bid_usd  # Sell at bid
                exit_opt_price = late_call.ask_usd   # Buy back at ask

                greeks = model.greeks(
                    smile_early.forward, atm_call.strike, smile_early.tau,
                    atm_iv, smile_early.rate, atm_call.option_type
                )
                inv = from_forward_greeks(
                    entry_opt_price, greeks, atm_call.index_price,
                    smile_early.forward, smile_early.tau, smile_early.rate
                )

                ntd = inv.net_transaction_delta

                # Delta hedge using forward move
                fwd_move = late_call.forward - smile_early.forward
                hedge_pnl = ntd * fwd_move  # Long hedge against short delta

                option_pnl = entry_opt_price - exit_opt_price
                fees = (entry_opt_price + exit_opt_price) * fee_rate
                net_pnl = option_pnl + hedge_pnl - fees

                trades.append({
                    "expiry": expiry,
                    "vrp": vrp,
                    "opt_pnl": option_pnl,
                    "hedge_pnl": hedge_pnl,
                    "fees": fees,
                    "net_pnl": net_pnl,
                })

    if not trades:
        print("No trades triggered. Try lowering --min-vrp.")
        return

    net_pnls = [t["net_pnl"] for t in trades]
    print(f"Total Trades Executed:   {len(trades)}")
    print(f"Win Rate:                {sum(1 for p in net_pnls if p > 0) / len(trades):.2%}")
    print(f"Cumulative Net P&L:      ${sum(net_pnls):,.2f}")
    print(f"Average P&L per Trade:   ${np.mean(net_pnls):,.2f}")
    print(f"P&L Volatility (StdDev): ${np.std(net_pnls):,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--min-vrp", type=float, default=4.0)
    parser.add_argument("--fee-bps", type=float, default=3.0)
    args = parser.parse_args()

    run_vrp_backtest(args.db, args.currency, args.min_vrp, args.fee_bps)