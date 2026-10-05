"""
Historical Options Market Making Backtest (Avellaneda-Stoikov Replay).
Replays snapshots in snapshots.db, posts passive SVI-anchored quotes,
and records passive fills when the market trades through our posted bids/asks.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Dict, List, Optional
import numpy as np

from deribit.chain import option_chain_from_snapshot
from deribit.pricing.black_scholes import Black76Model
from deribit.pricing.fees import calculate_future_fee, calculate_option_fee
from deribit.pricing.inverse import from_forward_greeks
from deribit.store import SnapshotStore
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.results import CalibratedSmile
from deribit.surface.simulator import QuotingSimulator, QuoteConfig
from deribit.surface.snapshot_loader import LoadedSnapshot, load_snapshot_observations


@dataclass
class InventoryPosition:
    instrument_name: str
    strike: float
    tau: float
    option_type: str
    contracts: float
    fill_price_usd: float
    vega: float


def run_market_making_backtest(
    db_path: str = "snapshots.db",
    currency: str = "BTC",
    min_snapshot_id: int = 28,
    limit: int = 200,
    number_of_starts: int = 4,
    min_hold_minutes: float = 10.0,
    max_hold_hours: float = 3.0,
    max_abs_k: float = 0.50,
):
    store = SnapshotStore(db_path)
    all_snaps = store.list_snapshots(currency=currency, limit=limit)
    snapshots = sorted([s for s in all_snaps if s.snapshot_id >= min_snapshot_id], key=lambda s: s.timestamp_ns)

    if len(snapshots) < 2:
        print(f"Need at least 2 snapshots with ID >= {min_snapshot_id}.")
        return

    config = QuoteConfig(
        min_half_width_usd=4.30,
        target_half_vol_points=1.5,
        rmse_weight=0.10,
        inventory_gamma=0.002,
        max_inventory_vega=1000.0,
    )
    simulator = QuotingSimulator(config=config)
    b76 = Black76Model()

    print("=" * 85)
    print("=== PRISM OPTIONS MARKET MAKING & DELTA-HEDGED INVENTORY BACKTEST ===")
    print("Model: SVI Fair Value Anchor + Avellaneda-Stoikov Inventory Shading")
    print(f"Replaying across {len(snapshots)} snapshots (#{snapshots[0].snapshot_id} -> #{snapshots[-1].snapshot_id})...")
    print("=" * 85)

    # In-memory caches to eliminate redundant parsing & calibration
    obs_cache: Dict[int, LoadedSnapshot] = {}
    smile_cache: Dict[tuple[int, int], CalibratedSmile] = {}

    def get_obs(snap_id: int) -> Optional[LoadedSnapshot]:
        if snap_id not in obs_cache:
            try:
                obs_cache[snap_id] = load_snapshot_observations(store, snap_id, max_abs_k=max_abs_k)
            except Exception:
                return None
        return obs_cache[snap_id]

    inventory: Dict[str, InventoryPosition] = {}
    futures_position: float = 0.0
    prev_index_px: float = 0.0

    total_quotes_posted = 0
    quotes_improving_book = 0
    evaluated_pairs = 0

    total_futures_pnl = 0.0
    total_futures_fees = 0.0
    total_opt_fees = 0.0

    for i in range(len(snapshots) - 1):
        s_early = snapshots[i]
        s_late = snapshots[i + 1]

        dt_sec = (s_late.timestamp_ns - s_early.timestamp_ns) / 1e9
        if not (min_hold_minutes * 60 <= dt_sec <= max_hold_hours * 3600):
            continue

        obs_early = get_obs(s_early.snapshot_id)
        obs_late = get_obs(s_late.snapshot_id)
        if not obs_early or not obs_late:
            continue

        sample_obs = next((obs[0] for obs in obs_early.observations_by_expiry.values() if obs), None)
        if not sample_obs:
            continue
        current_index_px = sample_obs.index_price

        evaluated_pairs += 1

        # Delta-Hedge PnL from previous step
        if futures_position != 0.0 and prev_index_px > 0.0:
            step_hedge_pnl = futures_position * (current_index_px - prev_index_px)
            total_futures_pnl += step_hedge_pnl

        prev_index_px = current_index_px

        # Live Progress Display
        print(
            f"[{i + 1}/{len(snapshots) - 1}] Replaying #{s_early.snapshot_id} -> #{s_late.snapshot_id} "
            f"| Spot: ${current_index_px:,.0f} | Fills: {len(simulator.fills):<3d} | Vega: {simulator.inventory_vega:+.1f}",
            end="\r",
        )

        for expiry, early_list in obs_early.observations_by_expiry.items():
            if expiry not in obs_late.observations_by_expiry or len(early_list) < 8:
                continue

            late_map = {o.instrument_name: o for o in obs_late.expiry(expiry)}

            # SVI Calibration with caching
            cache_key = (s_early.snapshot_id, expiry)
            if cache_key in smile_cache:
                smile = smile_cache[cache_key]
            else:
                try:
                    smile = calibrate_svi_slice(early_list, number_of_starts=number_of_starts, require_butterfly_free=True)
                    smile_cache[cache_key] = smile
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

                late_obs = late_map.get(obs.instrument_name)
                if not late_obs:
                    continue

                greeks = b76.greeks(smile.forward, obs.strike, smile.tau, obs.mid_iv, smile.rate, obs.option_type)

                vega_1vol = greeks.vega / 100.0
                # 1. Passive BUY Fill: Market crossed down to hit our bid
                if late_obs.ask_usd <= quote.shaded_bid_usd and quote.shaded_bid_usd > 0.0:
                    fill = simulator.process_fill(
                        quote=quote,
                        side="BUY",
                        fill_price_usd=quote.shaded_bid_usd,
                        timestamp_ns=s_late.timestamp_ns,
                        vega_1vol=vega_1vol,
                        contracts=0.5,
                    )
                    total_opt_fees += calculate_option_fee(quote.shaded_bid_usd, obs.index_price, contracts=0.5, is_taker=False)

                    pos = inventory.get(obs.instrument_name, InventoryPosition(
                        obs.instrument_name, obs.strike, smile.tau, obs.option_type, 0.0, quote.shaded_bid_usd, vega_1vol
                    ))
                    pos.contracts += 0.5
                    inventory[obs.instrument_name] = pos

                # 2. Passive SELL Fill: Market crossed up to lift our ask
                elif late_obs.bid_usd >= quote.shaded_ask_usd and quote.shaded_ask_usd > 0.0:
                    fill = simulator.process_fill(
                        quote=quote,
                        side="SELL",
                        fill_price_usd=quote.shaded_ask_usd,
                        timestamp_ns=s_late.timestamp_ns,
                        vega_1vol=vega_1vol,
                        contracts=0.5,
                    )
                    total_opt_fees += calculate_option_fee(quote.shaded_ask_usd, obs.index_price, contracts=0.5, is_taker=False)

                    pos = inventory.get(obs.instrument_name, InventoryPosition(
                        obs.instrument_name, obs.strike, smile.tau, obs.option_type, 0.0, quote.shaded_ask_usd, vega_1vol
                    ))
                    pos.contracts -= 0.5
                    inventory[obs.instrument_name] = pos

        # Rebalance Delta Hedge across total inventory
        total_port_delta = 0.0
        for pos in inventory.values():
            if pos.contracts != 0.0:
                g = b76.greeks(current_index_px, pos.strike, max(0.001, pos.tau), 0.55, 0.03, pos.option_type)
                total_port_delta += pos.contracts * g.delta

        desired_futures = -total_port_delta
        delta_change = desired_futures - futures_position
        if abs(delta_change) > 0.05:
            hedge_fee = calculate_future_fee(current_index_px, abs(delta_change), is_taker=True)
            total_futures_fees += hedge_fee
            futures_position = desired_futures

    print("\n" + "=" * 85)
    print("Marking terminal inventory to market...")

    # Terminal Inventory MTM
    last_snap = snapshots[-1]
    last_raw = store.load_snapshot(last_snap.snapshot_id)
    last_chain = option_chain_from_snapshot(last_raw)
    last_quote_map = {q.instrument_name: q for q in last_chain}

    inventory_mtm_pnl = 0.0
    for pos in inventory.values():
        if pos.contracts != 0.0:
            q = last_quote_map.get(pos.instrument_name)
            current_mid = (q.index_price * q.mid_coin) if q and q.mid_coin else pos.fill_price_usd
            pos_pnl = pos.contracts * (current_mid - pos.fill_price_usd)
            inventory_mtm_pnl += pos_pnl

    edge_captured = sum(f.edge_captured_usd for f in simulator.fills) if simulator.fills else 0.0
    total_fees = total_opt_fees + total_futures_fees
    net_strategy_pnl = inventory_mtm_pnl + total_futures_pnl - total_fees

    print(f"\nEvaluated Transitions:        {evaluated_pairs}")
    print(f"Total Quotes Posted:          {total_quotes_posted}")
    print(f"Quotes Improving Book:        {quotes_improving_book} ({quotes_improving_book / max(1, total_quotes_posted):.2%})")
    print(f"Total Passive Fills:          {len(simulator.fills)}")
    print(f"Final Inventory Vega:         {simulator.inventory_vega:+.2f}")
    print(f"Final Futures Delta Position: {futures_position:+.3f} BTC")
    print("-" * 55)
    print("--- P&L ATTRIBUTION ---")
    print(f"  Edge Captured at Fill:   ${edge_captured:+,.2f}  (Theoretical spread captured)")
    print(f"  Inventory Terminal MTM:  ${inventory_mtm_pnl:+,.2f}  (Actual Option P&L)")
    print(f"  Futures Delta Hedge P&L: ${total_futures_pnl:+,.2f}")
    print(f"  Total Fees (Opt + Fut):   ${total_fees:,.2f}")
    print("=" * 55)
    print(f"REALIZED NET STRATEGY P&L:     ${net_strategy_pnl:+,.2f}  (b + c - d)")

    if simulator.fills:
        print("\n--- SAMPLE RECENT FILLS ---")
        print(f"{'Side':<6} {'Instrument':<22} {'Fill Px':<11} {'Fair Value':<12} {'Edge Captured':<14}")
        print("-" * 65)
        for f in simulator.fills[-10:]:
            print(f"{f.side:<6} {f.instrument_name:<22} ${f.fill_price_usd:<10,.2f} ${f.tv_at_fill_usd:<11,.2f} +${f.edge_captured_usd:<13,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--min-snapshot-id", type=int, default=28)
    parser.add_argument("--limit", type=int, default=100, help="Number of snapshots to evaluate (default: 100)")
    parser.add_argument("--starts", type=int, default=4, help="SVI optimizer starts (default: 4 for speed)")
    args = parser.parse_args()

    run_market_making_backtest(
        args.db,
        args.currency,
        min_snapshot_id=args.min_snapshot_id,
        limit=args.limit,
        number_of_starts=args.starts,
    )