"""
SVI Dislocation Reversion Engine: Vega-Neutral, Delta-Hedged Strike Arbitrage.

Strategy Logic:
1. Fits SVI via Leave-One-Out (LOO) on each strike in an expiry.
2. Identifies strikes trading at significant spread-normalized residuals (|z| >= z_score_threshold).
3. Constructs a VEGA-NEUTRAL pair:
     Target Strike: 1.0 contract (Long if cheap, Short if rich)
     ATM Hedge Leg: beta = - (Vega_target / Vega_atm) contracts
4. Computes Net Transaction Delta (NTD) for the 2-leg option portfolio.
5. Hedges residual delta using the traded future/forward.
6. Evaluates PnL mark-to-market at the subsequent snapshot.
"""
from dataclasses import dataclass
import math
from typing import Dict, List, Optional

from deribit.chain import OptionQuote, option_chain_from_snapshot
from deribit.pricing.black_scholes import Black76Model
from deribit.pricing.inverse import from_forward_greeks
from deribit.store import SnapshotStore
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.snapshot_loader import load_snapshot_observations


@dataclass(frozen=True)
class DislocationTrade:
    entry_snapshot_id: int
    exit_snapshot_id: int
    expiry: int
    target_instrument: str
    target_strike: float
    target_type: str
    atm_instrument: str
    side: str  # 'BUY_DISLOCATED' or 'SELL_DISLOCATED'
    target_entry_usd: float
    target_exit_usd: float
    atm_entry_usd: float
    atm_exit_usd: float
    atm_hedge_ratio: float
    residual_z_score: float
    target_pnl_usd: float
    atm_hedge_pnl_usd: float
    delta_hedge_pnl_usd: float
    fee_drag_usd: float
    net_pnl_usd: float


class SVIReversionStrategy:
    def __init__(
        self,
        db_path: str = "snapshots.db",
        currency: str = "BTC",
        z_score_threshold: float = 1.0,  # Mispricing must exceed 1.0x half-spread
        entry_threshold_bps: float | None = None,
        taker_fee_bps: float = 3.0,
        fut_fee_bps: float = 1.5,
        max_abs_k: float = 0.50,         # Focus on liquid near-the-money range
        use_leave_one_out: bool = True,
        **kwargs
    ):
        self.store = SnapshotStore(db_path)
        self.currency = currency
        self.z_score_threshold = z_score_threshold
        self.taker_fee_rate = taker_fee_bps / 10000.0
        self.fut_fee_rate = fut_fee_bps / 10000.0
        self.max_abs_k = max_abs_k
        self.model = Black76Model()
        self.use_leave_one_out = use_leave_one_out

    def run_backtest(self, snapshot_limit: int = 500) -> List[DislocationTrade]:
        snapshots = self.store.list_snapshots(currency=self.currency, limit=snapshot_limit)
        snapshots = sorted(snapshots, key=lambda s: s.timestamp_ns)

        if len(snapshots) < 2:
            return []

        closed_trades: List[DislocationTrade] = []

        for i in range(len(snapshots) - 1):
            snap_early = snapshots[i]
            snap_late = snapshots[i + 1]

            print(f"[{i + 1}/{len(snapshots) - 1}] Evaluating Snapshot #{snap_early.snapshot_id} -> #{snap_late.snapshot_id}...")
            try:
                obs_early = load_snapshot_observations(self.store, snap_early.snapshot_id, max_abs_k=self.max_abs_k)
                snap_late_data = self.store.load_snapshot(snap_late.snapshot_id)
                late_chain = option_chain_from_snapshot(snap_late_data)
                late_quote_map = {q.instrument_name: q for q in late_chain}
            except Exception:
                continue

            for expiry, obs_list in obs_early.observations_by_expiry.items():
                if len(obs_list) < 8:
                    continue

                # Locate the most liquid ATM reference quote in early snapshot
                forward_early = obs_list[0].forward
                atm_candidate = min(obs_list, key=lambda o: abs(o.strike - forward_early))
                late_atm = late_quote_map.get(atm_candidate.instrument_name)

                if not late_atm or not late_atm.mid_usd or atm_candidate.mid_usd is None:
                    continue

                for target_obs in obs_list:
                    # Do not pair ATM against itself
                    if target_obs.source_row_id == atm_candidate.source_row_id:
                        continue

                    # Leave-One-Out (LOO) Calibration
                    loo_slice = [o for o in obs_list if o.source_row_id != target_obs.source_row_id]
                    try:
                        smile = calibrate_svi_slice(loo_slice, require_butterfly_free=True)
                    except Exception:
                        continue

                    # Evaluate SVI Fair Value & Half-Spread Normalized Residual
                    k = target_obs.log_moneyness
                    tv_w = float(smile.parameters.total_variance(k))
                    tv_iv = math.sqrt(max(1e-12, tv_w) / smile.tau)
                    tv_usd = self.model.price(
                        smile.forward, target_obs.strike, smile.tau, tv_iv, smile.rate, target_obs.option_type
                    )

                    half_spread_usd = 0.5 * (target_obs.ask_usd - target_obs.bid_usd)
                    if half_spread_usd <= 0.0:
                        continue

                    price_residual = target_obs.mid_usd - tv_usd
                    z_score = price_residual / half_spread_usd

                    # Must exceed our dislocation threshold
                    if abs(z_score) < self.z_score_threshold:
                        continue

                    late_target = late_quote_map.get(target_obs.instrument_name)
                    if not late_target or not late_target.mid_usd:
                        continue

                    # Calculate Greeks for Target & ATM leg
                    g_target = self.model.greeks(
                        smile.forward, target_obs.strike, smile.tau, target_obs.mid_iv, smile.rate, target_obs.option_type
                    )
                    g_atm = self.model.greeks(
                        smile.forward, atm_candidate.strike, smile.tau, atm_candidate.mid_iv, smile.rate, atm_candidate.option_type
                    )

                    if g_atm.vega < 1e-4:
                        continue

                    # VEGA HEDGE RATIO: beta = - (Vega_target / Vega_atm)
                    vega_ratio = g_target.vega / g_atm.vega

                    # Signal Direction:
                    # If target is CHEAP (z_score < -threshold): BUY Target, SELL ATM hedge
                    # If target is RICH  (z_score > +threshold): SELL Target, BUY ATM hedge
                    is_cheap = z_score < 0.0

                    if is_cheap:
                        # BUY Target at Ask, SELL ATM at Bid
                        target_entry = target_obs.ask_usd
                        atm_entry = atm_candidate.bid_usd
                        target_exit = late_target.index_price * (late_target.bid_coin or 0.0)
                        atm_exit = late_atm.index_price * (late_atm.ask_coin or 0.0)
                        side_mult = 1.0
                    else:
                        # SELL Target at Bid, BUY ATM at Ask
                        target_entry = target_obs.bid_usd
                        atm_entry = atm_candidate.ask_usd
                        target_exit = late_target.index_price * (late_target.ask_coin or 0.0)
                        atm_exit = late_atm.index_price * (late_atm.bid_coin or 0.0)
                        side_mult = -1.0

                    if target_entry <= 0.0 or atm_entry <= 0.0 or target_exit <= 0.0 or atm_exit <= 0.0:
                        continue

                    # Target Option PnL
                    target_pnl = side_mult * (target_exit - target_entry)

                    # ATM Option PnL (holds -vega_ratio contracts)
                    atm_contracts = -side_mult * vega_ratio
                    atm_pnl = atm_contracts * (atm_exit - atm_entry)

                    # Net Transaction Delta (NTD) for each leg using inverse.py
                    inv_target = from_forward_greeks(
                        target_entry, g_target, target_obs.index_price, smile.forward, smile.tau, smile.rate
                    )
                    inv_atm = from_forward_greeks(
                        atm_entry, g_atm, atm_candidate.index_price, smile.forward, smile.tau, smile.rate
                    )

                    portfolio_ntd = (side_mult * inv_target.net_transaction_delta) + (atm_contracts * inv_atm.net_transaction_delta)

                    # Delta hedge with Forward/Future
                    forward_late = late_target.api_forward or late_target.index_price
                    spot_change = forward_late - smile.forward
                    delta_hedge_pnl = -portfolio_ntd * spot_change

                    # Taker Fees
                    opt_fees = (
                        (target_entry + target_exit) * self.taker_fee_rate
                        + abs(atm_contracts) * (atm_entry + atm_exit) * self.taker_fee_rate
                    )
                    hedge_fees = abs(portfolio_ntd) * (smile.forward + forward_late) * 0.5 * self.fut_fee_rate
                    total_fees = opt_fees + hedge_fees

                    net_pnl = target_pnl + atm_pnl + delta_hedge_pnl - total_fees

                    closed_trades.append(
                        DislocationTrade(
                            entry_snapshot_id=snap_early.snapshot_id,
                            exit_snapshot_id=snap_late.snapshot_id,
                            expiry=expiry,
                            target_instrument=target_obs.instrument_name,
                            target_strike=target_obs.strike,
                            target_type=target_obs.option_type,
                            atm_instrument=atm_candidate.instrument_name,
                            side="BUY_DISLOCATED" if is_cheap else "SELL_DISLOCATED",
                            target_entry_usd=target_entry,
                            target_exit_usd=target_exit,
                            atm_entry_usd=atm_entry,
                            atm_exit_usd=atm_exit,
                            atm_hedge_ratio=atm_contracts,
                            residual_z_score=z_score,
                            target_pnl_usd=target_pnl,
                            atm_hedge_pnl_usd=atm_pnl,
                            delta_hedge_pnl_usd=delta_hedge_pnl,
                            fee_drag_usd=total_fees,
                            net_pnl_usd=net_pnl,
                        )
                    )

        return closed_trades