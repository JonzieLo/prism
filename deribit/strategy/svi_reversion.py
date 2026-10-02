"""
SVI Dislocation Reversion Engine: Vega-Neutral, Delta-Hedged Strike Arbitrage.

Strategy Logic:
1. Fits base SVI smile once per expiry.
2. Screens for potential dislocations using vectorized spread-normalized residuals.
3. Only executes Leave-One-Out (LOO) calibration on candidate outlier strikes.
4. Caches loaded snapshot observations in memory to eliminate redundant parsing.
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
        max_abs_k: float = 0.50,
        use_leave_one_out: bool = True,
        **kwargs,
    ):
        self.store = SnapshotStore(db_path)
        self.currency = currency
        self.z_score_threshold = z_score_threshold
        self.taker_fee_rate = taker_fee_bps / 10000.0
        self.fut_fee_rate = fut_fee_bps / 10000.0
        self.max_abs_k = max_abs_k
        self.use_leave_one_out = use_leave_one_out
        self.model = Black76Model()
        self._obs_cache: dict[int, any] = {}

    def _get_snapshot_obs(self, snap_id: int):
        if snap_id not in self._obs_cache:
            self._obs_cache[snap_id] = load_snapshot_observations(
                self.store, snap_id, max_abs_k=self.max_abs_k
            )
        return self._obs_cache[snap_id]

    def run_backtest(self, snapshot_limit: int = 500) -> List[DislocationTrade]:
        snapshots = self.store.list_snapshots(currency=self.currency, limit=snapshot_limit)
        snapshots = sorted(snapshots, key=lambda s: s.timestamp_ns)

        if len(snapshots) < 2:
            return []

        closed_trades: List[DislocationTrade] = []

        for i in range(len(snapshots) - 1):
            snap_early = snapshots[i]
            snap_late = snapshots[i + 1]

            dt_seconds = (snap_late.timestamp_ns - snap_early.timestamp_ns) / 1e9
            if not (600 <= dt_seconds <= 10800):
                continue
            print(f"[{i + 1}/{len(snapshots) - 1}] Evaluating Snapshot #{snap_early.snapshot_id} -> #{snap_late.snapshot_id}...")

            try:
                obs_early = self._get_snapshot_obs(snap_early.snapshot_id)
                snap_late_data = self.store.load_snapshot(snap_late.snapshot_id)
                late_chain = option_chain_from_snapshot(snap_late_data)
                late_quote_map = {q.instrument_name: q for q in late_chain}
            except Exception:
                continue

            for expiry, obs_list in obs_early.observations_by_expiry.items():
                if len(obs_list) < 8:
                    continue

                # Locate ATM reference quote (closest to forward)
                forward_early = obs_list[0].forward
                atm_candidate = min(obs_list, key=lambda o: abs(o.strike - forward_early))
                late_atm = late_quote_map.get(atm_candidate.instrument_name)

                if not late_atm or not late_atm.mid_usd or atm_candidate.mid_usd is None:
                    continue

                # --- STEP 1: FAST BASELINE CALIBRATION (Fitted once per expiry) ---
                try:
                    base_smile = calibrate_svi_slice(obs_list, number_of_starts=8, require_butterfly_free=True)
                except Exception:
                    continue

                # Precompute ATM vega
                g_atm = self.model.greeks(
                    base_smile.forward, atm_candidate.strike, base_smile.tau,
                    atm_candidate.mid_iv, base_smile.rate, atm_candidate.option_type
                )
                if g_atm.vega < 1e-4:
                    continue

                for target_obs in obs_list:
                    if target_obs.source_row_id == atm_candidate.source_row_id:
                        continue

                    # --- STEP 2: SCREENING FILTER ---
                    # Check baseline residual first before triggering expensive LOO
                    k_t = target_obs.log_moneyness
                    base_w = float(base_smile.parameters.total_variance(k_t))
                    base_iv = math.sqrt(max(1e-12, base_w) / base_smile.tau)
                    base_usd = self.model.price(
                        base_smile.forward, target_obs.strike, base_smile.tau,
                        base_iv, base_smile.rate, target_obs.option_type
                    )

                    half_spread_usd = 0.5 * (target_obs.ask_usd - target_obs.bid_usd)
                    if half_spread_usd <= 0.0:
                        continue

                    base_z = (target_obs.mid_usd - base_usd) / half_spread_usd

                    # If the strike sits tightly on the curve (|z| < 0.8x threshold), skip LOO!
                    if abs(base_z) < (self.z_score_threshold * 0.8):
                        continue

                    # --- STEP 3: VERIFY CANDIDATE VIA LEAVE-ONE-OUT (LOO) ---
                    if self.use_leave_one_out:
                        loo_slice = [o for o in obs_list if o.source_row_id != target_obs.source_row_id]
                        try:
                            smile = calibrate_svi_slice(loo_slice, number_of_starts=4, require_butterfly_free=True)
                        except Exception:
                            continue
                    else:
                        smile = base_smile

                    # Evaluate exact LOO fair value
                    tv_w = float(smile.parameters.total_variance(k_t))
                    tv_iv = math.sqrt(max(1e-12, tv_w) / smile.tau)
                    tv_usd = self.model.price(
                        smile.forward, target_obs.strike, smile.tau,
                        tv_iv, smile.rate, target_obs.option_type
                    )

                    price_residual = target_obs.mid_usd - tv_usd
                    z_score = price_residual / half_spread_usd

                    if abs(z_score) < self.z_score_threshold:
                        continue

                    # Verify exit quote in late snapshot
                    late_target = late_quote_map.get(target_obs.instrument_name)
                    if not late_target or not late_target.mid_usd:
                        continue

                    # Target Greeks & Vega-neutral hedge ratio
                    g_target = self.model.greeks(
                        smile.forward, target_obs.strike, smile.tau,
                        target_obs.mid_iv, smile.rate, target_obs.option_type
                    )
                    vega_ratio = g_target.vega / g_atm.vega

                    is_cheap = z_score < 0.0
                    # if is_cheap:
                    #     target_entry = target_obs.ask_usd
                    #     atm_entry = atm_candidate.bid_usd
                    #     target_exit = late_target.index_price * (late_target.bid_coin or 0.0)
                    #     atm_exit = late_atm.index_price * (late_atm.ask_coin or 0.0)
                    #     side_mult = 1.0
                    # else:
                    #     target_entry = target_obs.bid_usd
                    #     atm_entry = atm_candidate.ask_usd
                    #     target_exit = late_target.index_price * (late_target.ask_coin or 0.0)
                    #     atm_exit = late_atm.index_price * (late_atm.bid_coin or 0.0)
                    #     side_mult = -1.0

                    target_entry = target_obs.mid_usd
                    atm_entry = atm_candidate.mid_usd
                    
                    # Next snapshot midpoints
                    target_exit = late_target.index_price * late_target.mid_coin
                    atm_exit = late_atm.index_price * late_atm.mid_coin
                    side_mult = 1.0 if is_cheap else -1.0

                    if target_entry <= 0.0 or atm_entry <= 0.0 or target_exit <= 0.0 or atm_exit <= 0.0:
                        continue

                    target_pnl = side_mult * (target_exit - target_entry)
                    atm_contracts = -side_mult * vega_ratio
                    atm_pnl = atm_contracts * (atm_exit - atm_entry)

                    # Net Transaction Delta (NTD)
                    inv_target = from_forward_greeks(
                        target_entry, g_target, target_obs.index_price, smile.forward, smile.tau, smile.rate
                    )
                    inv_atm = from_forward_greeks(
                        atm_entry, g_atm, atm_candidate.index_price, smile.forward, smile.tau, smile.rate
                    )

                    portfolio_ntd = (side_mult * inv_target.net_transaction_delta) + (atm_contracts * inv_atm.net_transaction_delta)

                    forward_late = late_target.api_forward or late_target.index_price
                    spot_change = forward_late - smile.forward
                    delta_hedge_pnl = -portfolio_ntd * spot_change

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