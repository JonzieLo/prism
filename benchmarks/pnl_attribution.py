import argparse
from deribit.store import SnapshotStore, fetch_and_save_snapshot
from deribit.surface.snapshot_loader import load_snapshot_observations
from deribit.surface.calibration import calibrate_svi_slice
from deribit.pricing.black_scholes import Black76Model


def run_pnl_attribution(snapshot_a_id: int, snapshot_b_id: int, db_path: str = "snapshots.db"):
    store = SnapshotStore(db_path)
    loaded_a = load_snapshot_observations(store, snapshot_a_id)
    loaded_b = load_snapshot_observations(store, snapshot_b_id)

    # Time elapsed in years
    dt_seconds = (loaded_b.timestamp_ns - loaded_a.timestamp_ns) / 1e9
    dt_years = dt_seconds / (365.25 * 86400)
    
    model = Black76Model()

    print("=" * 80)
    print(f"=== P&L ATTRIBUTION: Snapshot #{snapshot_a_id} -> Snapshot #{snapshot_b_id} ===")
    print(f"=== Time Elapsed: {dt_seconds:.1f}s ({dt_years:.6f} years) ===")
    print("=" * 80)

    for expiry, obs_a_list in loaded_a.observations_by_expiry.items():
        if expiry not in loaded_b.observations_by_expiry:
            continue

        try:
            smile_a = calibrate_svi_slice(obs_a_list)
        except Exception:
            continue

        obs_b_map = {o.instrument_name: o for o in loaded_b.expiry(expiry)}

        for obs_a in obs_a_list:
            obs_b = obs_b_map.get(obs_a.instrument_name)
            if not obs_b or obs_b.mid_usd is None or obs_a.mid_usd is None:
                continue

            # Pricing inputs
            F_a, F_b = smile_a.forward, obs_b.forward
            dS = F_b - F_a
            
            k_a = obs_a.log_moneyness
            vol_a = float(smile_a.parameters.implied_vol(k_a, smile_a.tau))
            vol_b = obs_b.mid_iv
            d_vol = vol_b - vol_a

            # Baseline option price and Greeks at Snapshot A
            p_a = model.price(F_a, obs_a.strike, obs_a.tau, vol_a, obs_a.rate, obs_a.option_type)
            greeks = model.greeks(F_a, obs_a.strike, obs_a.tau, vol_a, obs_a.rate, obs_a.option_type)

            # Observed PnL
            p_b = obs_b.mid_usd
            actual_pnl = p_b - p_a

            # Attributed Taylor components
            delta_pnl = greeks.delta * dS
            gamma_pnl = 0.5 * greeks.gamma * (dS ** 2)
            vega_pnl = greeks.vega * d_vol
            theta_pnl = greeks.theta * dt_years

            explained_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
            residual = actual_pnl - explained_pnl

            print(f"\nInstrument: {obs_a.instrument_name}")
            print(f"  Actual PnL:   ${actual_pnl:+8.2f}")
            print(f"  Delta PnL:    ${delta_pnl:+8.2f} (dS = ${dS:+.2f})")
            print(f"  Gamma PnL:    ${gamma_pnl:+8.2f}")
            print(f"  Vega PnL:     ${vega_pnl:+8.2f} (dVol = {d_vol:+.4f})")
            print(f"  Theta PnL:    ${theta_pnl:+8.2f}")
            print(f"  Residual:     ${residual:+8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run Milestone M5 P&L Attribution.")
    parser.add_argument("--snap-a", type=int, help="Initial snapshot ID")
    parser.add_argument("--snap-b", type=int, help="Ending snapshot ID")
    parser.add_argument("--fetch-b", action="store_true", help="Capture a fresh live Snapshot B right now")
    parser.add_argument("--prod", action="store_true", help="Use Deribit Production (mainnet) for live fetch")
    parser.add_argument("--currency", default="BTC", help="Base currency (BTC/ETH)")
    parser.add_argument("--db", default="snapshots.db", help="SQLite database path")
    args = parser.parse_args()

    store = SnapshotStore(args.db)

    latest_id = store.latest_snapshot_id(args.currency)
    snap_a_id = args.snap_a or (latest_id - 1 if latest_id and latest_id > 1 else None)

    if not snap_a_id:
        print("Need at least 2 snapshots in the database to run attribution.")
        return

    if args.fetch_b:
        print(f"Capturing fresh live Snapshot B from Deribit ({'Prod' if args.prod else 'Testnet'})...")
        snap_b_id = fetch_and_save_snapshot(store, currency=args.currency, testnet=not args.prod)
    else:
        snap_b_id = args.snap_b or latest_id

    if snap_a_id == snap_b_id:
        print("Snapshot A and Snapshot B are identical. Please specify distinct snapshot IDs.")
        return

    run_pnl_attribution(snap_a_id, snap_b_id, args.db)


if __name__ == "__main__":
    main()