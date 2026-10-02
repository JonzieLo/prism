"""
Historical Options Market Making Backtest (Avellaneda-Stoikov Replay).
Replays snapshots in snapshots.db, posts passive SVI-anchored quotes,
and records passive fills when the market trades through our posted bids/asks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import numpy as np

from deribit.pricing.black_scholes import Black76Model
from deribit.store import SnapshotStore
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.simulator import QuotingSimulator, QuoteConfig
from deribit.surface.snapshot_loader import load_snapshot_observations


def run_market_making_backtest(
    db_path: str = "snapshots.db",
    currency: str = "BTC",
    min_hold_minutes: float = 10.0,
    max_hold_hours: float = 3.0,
    max_abs_k: float = 0.50,
):
    store = SnapshotStore(db_path)
    snapshots = store.list_snapshots(currency=currency, limit=500)
    snapshots = sorted(snapshots, key=lambda s: s.timestamp_ns)

    if len(snapshots) < 2:
        print("Need at least 2 snapshots to replay market making.")
        return

    simulator = QuotingSimulator(config=QuoteConfig())
    b76 = Black76Model()

    print("=" * 80)
    print("=== PRISM OPTIONS MARKET MAKING BACKTEST ===")
    print("Model: SVI Fair Value Anchor + Avellaneda-Stoikov Inventory Shading")
    print(f"Replaying across {len(snapshots)} snapshots...")
    print("=" * 80)

    total_quotes_posted = 0
    quotes_improving_book = 0
    evaluated_pairs = 0

    for i in range(len(snapshots) - 1):
        s_early = snapshots[i]
        s_late = snapshots[i + 1]

        # Valid holding horizon filter (between 10m and 3h)
        dt_sec = (s_late.timestamp_ns - s_early.timestamp_ns) / 1e9
        if not (min_hold_minutes * 60 <= dt_sec <= max_hold_hours * 3600):
            continue

        try:
            obs_early = load_snapshot_observations(store, s_early.snapshot_id, max_abs_k=max_abs_k)
            obs_late = load_snapshot_observations(store, s_late.snapshot_id, max_abs_k=max_abs_k)
        except Exception:
            continue

        evaluated_pairs += 1

        for expiry, early_list in obs_early.observations_by_expiry.items():
            if expiry not in obs_late.observations_by_expiry or len(early_list) < 8:
                continue

            late_map = {o.instrument_name: o for o in obs_late.expiry(expiry)}

            # Calibrate SVI smile
            try:
                smile = calibrate_svi_slice(early_list, number_of_starts=8, require_butterfly_free=True)
            except Exception:
                continue

            for obs in early_list:
                quote = simulator.generate_quote(
                    smile=smile,
                    strike=obs.strike,
                    option_type=obs.option_type,
                    instrument_name=obs.instrument_name,
                    market_bid_usd=obs.bid_usd,
                    market_ask_usd=obs.ask_usd,
                )
                if quote is None:
                    continue

                total_quotes_posted += 1
                if quote.bid_relation == "IMPROVED" or quote.ask_relation == "IMPROVED":
                    quotes_improving_book += 1

                # Check if market in Snapshot B traded through our posted quote
                late_obs = late_map.get(obs.instrument_name)
                if not late_obs:
                    continue

                greeks = b76.greeks(smile.forward, obs.strike, smile.tau, obs.mid_iv, smile.rate, obs.option_type)

                # Passive BUY Fill: Market sold into our posted bid
                if late_obs.bid_usd <= quote.shaded_bid_usd and quote.shaded_bid_usd > 0:
                    simulator.process_fill(
                        quote=quote,
                        side="BUY",
                        fill_price_usd=quote.shaded_bid_usd,
                        timestamp_ns=s_late.timestamp_ns,
                        vega=greeks.vega,
                        contracts=0.5,
                    )

                # Passive SELL Fill: Market bought our posted ask
                elif late_obs.ask_usd >= quote.shaded_ask_usd and quote.shaded_ask_usd > 0:
                    simulator.process_fill(
                        quote=quote,
                        side="SELL",
                        fill_price_usd=quote.shaded_ask_usd,
                        timestamp_ns=s_late.timestamp_ns,
                        vega=greeks.vega,
                        contracts=0.5,
                    )

    print(f"\nEvaluated Transitions:      {evaluated_pairs}")
    print(f"Total Quotes Posted:        {total_quotes_posted}")
    print(f"Quotes Improving Book:      {quotes_improving_book} ({quotes_improving_book / max(1, total_quotes_posted):.2%})")
    print(f"Passive Fills Captured:     {len(simulator.fills)}")

    if simulator.fills:
        edges = [f.edge_captured_usd for f in simulator.fills]
        total_edge = sum(edges)
        print(f"Cumulative Edge Captured:   ${total_edge:,.2f}")
        print(f"Average Edge per Fill:      ${np.mean(edges):,.2f}")
        print(f"Final Inventory Vega:       {simulator.inventory_vega:+.2f}")
        print("\n--- SAMPLE PASSIVE FILLS ---")
        print(f"{'Side':<6} {'Instrument':<22} {'Fill Px':<11} {'Fair Value':<12} {'Edge Captured':<14}")
        print("-" * 65)
        for f in simulator.fills[:8]:
            print(f"{f.side:<6} {f.instrument_name:<22} ${f.fill_price_usd:<10,.2f} ${f.tv_at_fill_usd:<11,.2f} +${f.edge_captured_usd:<13,.2f}")
    else:
        print("\nNo passive fills captured in this historical replay window.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--currency", default="BTC")
    args = parser.parse_args()

    run_market_making_backtest(args.db, args.currency)