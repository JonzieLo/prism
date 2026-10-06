"""
SVI Dislocation Reversion Engine: Vega-Neutral, Delta-Hedged Strike Arbitrage.
"""
from dataclasses import dataclass
import math
from typing import Dict, List, Optional

from deribit.chain import OptionQuote, option_chain_from_snapshot
from deribit.pricing.black_scholes import Black76Model
from deribit.pricing.fees import calculate_future_fee,calculate_option_combo_fee
from deribit.pricing.inverse import from_forward_greeks
from deribit.store import SnapshotStore
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.snapshot_loader import load_snapshot_observations


@dataclass
class OpenPosition:
    entry_snapshot_id: int
    entry_timestamp_ns: int
    expiry: int
    target_instrument: str
    target_strike: float
    target_type: str
    atm_instrument: str
    side: str  # 'BUY_DISLOCATED' or 'SELL_DISLOCATED'
    target_entry_usd: float
    atm_entry_usd: float
    atm_contracts: float
    portfolio_ntd: float
    entry_forward: float
    initial_z_score: float
    entry_fee_usd: float
    snapshots_held: int = 0


@dataclass(frozen=True)
class DislocationTrade:
    entry_snapshot_id: int
    exit_snapshot_id: int
    entry_timestamp_ns: int
    exit_timestamp_ns: int
    expiry: int
    target_instrument: str
    target_strike: float
    target_type: str
    atm_instrument: str
    side: str
    target_entry_usd: float
    target_exit_usd: float
    atm_entry_usd: float
    atm_exit_usd: float
    atm_hedge_ratio: float
    residual_z_score: float
    snapshots_held: int
    exit_reason: str  # 'REVERTED' or 'TIME_STOP'
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
        z_score_threshold: float = 2.5,  # Mispricing must exceed 2.5x half-spread
        exit_z_threshold: float = 0.5,
        max_holding_snapshots: int = 8, 
        min_edge_fee_multiple: float = 0.5,
        max_abs_k: float = 0.50,
        use_leave_one_out: bool = True,
        use_midpoint: bool = True,           # True: evaluate at mid; False: cross bid/ask spreads
        **kwargs,
    ):
        self.store = SnapshotStore(db_path)
        self.currency = currency
        self.z_score_threshold = z_score_threshold
        self.exit_z_threshold = exit_z_threshold
        self.max_holding_snapshots = max_holding_snapshots
        self.min_edge_fee_multiple = min_edge_fee_multiple
        self.max_abs_k = max_abs_k
        self.use_leave_one_out = use_leave_one_out
        self.use_midpoint = use_midpoint
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

        open_positions: Dict[str, OpenPosition] = {}
        closed_trades: List[DislocationTrade] = []

        for i in range(len(snapshots) - 1):
            snap_curr = snapshots[i]
            snap_next = snapshots[i + 1]

            dt_seconds = (snap_next.timestamp_ns - snap_curr.timestamp_ns) / 1e9
            if not (600 <= dt_seconds <= 10800):
                continue
            print(f"[{i + 1}/{len(snapshots) - 1}] Evaluating Snapshot #{snap_curr.snapshot_id} -> #{snap_next.snapshot_id}...")

            try:
                obs_curr = self._get_snapshot_obs(snap_curr.snapshot_id)
                next_raw = self.store.load_snapshot(snap_next.snapshot_id)
                next_chain = option_chain_from_snapshot(next_raw)
                next_quote_map = {q.instrument_name: q for q in next_chain}
            except Exception:
                continue

            for inst_name in list(open_positions.keys()):
                pos = open_positions[inst_name]
                pos.snapshots_held += 1

                late_target = next_quote_map.get(pos.target_instrument)
                late_atm = next_quote_map.get(pos.atm_instrument)

                if not late_target or not late_atm or not late_target.mid_usd or not late_atm.mid_usd:
                    continue

                late_idx = late_target.index_price
                if self.use_midpoint:
                    target_exit_px = late_idx * (late_target.mid_coin or 0.0)
                    atm_exit_px = late_idx * (late_atm.mid_coin or 0.0)
                else:
                    if pos.side == "BUY_DISLOCATED":
                        target_exit_px = late_idx * (late_target.bid_coin or 0.0)
                        atm_exit_px = late_idx * (late_atm.ask_coin or 0.0)
                    else:
                        target_exit_px = late_idx * (late_target.ask_coin or 0.0)
                        atm_exit_px = late_idx * (late_atm.bid_coin or 0.0)


                current_residual = abs(target_exit_px - pos.target_entry_usd)
                has_reverted = pos.snapshots_held >= 2 and (pos.snapshots_held >= self.max_holding_snapshots or current_residual < 10.0)
                time_stopped = pos.snapshots_held >= self.max_holding_snapshots

                if has_reverted or time_stopped:
                    # CLOSE POSITION
                    side_mult = 1.0 if pos.side == "BUY_DISLOCATED" else -1.0
                    target_pnl = side_mult * (target_exit_px - pos.target_entry_usd)
                    atm_pnl = pos.atm_contracts * (atm_exit_px - pos.atm_entry_usd)

                    fwd_late = late_target.api_forward or late_idx
                    spot_change = fwd_late - pos.entry_forward
                    delta_hedge_pnl = -pos.portfolio_ntd * spot_change

                    # Calculate exit fee
                    if pos.side == "BUY_DISLOCATED":
                        exit_buy_legs = [(atm_exit_px, abs(pos.atm_contracts))]
                        exit_sell_legs = [(target_exit_px, 1.0)]
                    else:
                        exit_buy_legs = [(target_exit_px, 1.0)]
                        exit_sell_legs = [(atm_exit_px, abs(pos.atm_contracts))]

                    exit_opt_fee = calculate_option_combo_fee(
                        buy_legs=exit_buy_legs, sell_legs=exit_sell_legs, index_price=late_idx, is_taker=True
                    )
                    exit_hedge_fee = calculate_future_fee(
                        future_price_usd=fwd_late, contracts=abs(pos.portfolio_ntd), is_taker=True
                    )
                    total_trade_fees = pos.entry_fee_usd + exit_opt_fee + exit_hedge_fee
                    net_pnl = target_pnl + atm_pnl + delta_hedge_pnl - total_trade_fees

                    closed_trades.append(
                        DislocationTrade(
                            entry_snapshot_id=pos.entry_snapshot_id,
                            exit_snapshot_id=snap_next.snapshot_id,
                            entry_timestamp_ns=pos.entry_timestamp_ns,
                            exit_timestamp_ns=snap_next.timestamp_ns,
                            expiry=pos.expiry,
                            target_instrument=pos.target_instrument,
                            target_strike=pos.target_strike,
                            target_type=pos.target_type,
                            atm_instrument=pos.atm_instrument,
                            side=pos.side,
                            target_entry_usd=pos.target_entry_usd,
                            target_exit_usd=target_exit_px,
                            atm_entry_usd=pos.atm_entry_usd,
                            atm_exit_usd=atm_exit_px,
                            atm_hedge_ratio=pos.atm_contracts,
                            residual_z_score=pos.initial_z_score,
                            snapshots_held=pos.snapshots_held,
                            exit_reason="REVERTED" if has_reverted and not time_stopped else "TIME_STOP",
                            target_pnl_usd=target_pnl,
                            atm_hedge_pnl_usd=atm_pnl,
                            delta_hedge_pnl_usd=delta_hedge_pnl,
                            fee_drag_usd=total_trade_fees,
                            net_pnl_usd=net_pnl,
                        )
                    )
                    del open_positions[inst_name]

            for expiry, obs_list in obs_curr.observations_by_expiry.items():
                if len(obs_list) < 8:
                    continue

                fwd_early = obs_list[0].forward
                atm_cand = min(obs_list, key=lambda o: abs(o.strike - fwd_early))
                if atm_cand.mid_usd is None:
                    continue

                try:
                    base_smile = calibrate_svi_slice(obs_list, number_of_starts=8, require_butterfly_free=True)
                except Exception:
                    continue

                g_atm = self.model.greeks(
                    base_smile.forward, atm_cand.strike, base_smile.tau,
                    atm_cand.mid_iv, base_smile.rate, atm_cand.option_type
                )
                if g_atm.vega < 1e-4:
                    continue

                for target_obs in obs_list:
                    # Skip if already holding this strike or if ATM
                    if target_obs.instrument_name in open_positions or target_obs.strike == atm_cand.strike:
                        continue

                    k_t = target_obs.log_moneyness
                    base_w = float(base_smile.parameters.total_variance(k_t))
                    base_iv = math.sqrt(max(1e-12, base_w) / base_smile.tau)
                    base_usd = self.model.price(
                        base_smile.forward, target_obs.strike, base_smile.tau,
                        base_iv, base_smile.rate, target_obs.option_type
                    )

                    half_spread = 0.5 * (target_obs.ask_usd - target_obs.bid_usd)
                    if half_spread <= 0.0:
                        continue

                    price_residual = target_obs.mid_usd - base_usd
                    z_score = price_residual / half_spread

                    if abs(z_score) < self.z_score_threshold:
                        continue

                    if self.use_leave_one_out:
                        loo_slice = [o for o in obs_list if o.source_row_id != target_obs.source_row_id]
                        try:
                            smile = calibrate_svi_slice(loo_slice, number_of_starts=4, require_butterfly_free=True)
                        except Exception:
                            continue
                    else:
                        smile = base_smile

                    tv_w = float(smile.parameters.total_variance(k_t))
                    tv_iv = math.sqrt(max(1e-12, tv_w) / smile.tau)
                    tv_usd = self.model.price(
                        smile.forward, target_obs.strike, smile.tau,
                        tv_iv, smile.rate, target_obs.option_type
                    )

                    price_res_loo = target_obs.mid_usd - tv_usd
                    z_loo = price_res_loo / half_spread

                    if abs(z_loo) < self.z_score_threshold:
                        continue

                    # Execution Prices
                    is_cheap = z_loo < 0.0
                    early_idx = target_obs.index_price

                    if self.use_midpoint:
                        target_entry = target_obs.mid_usd
                        atm_entry = atm_cand.mid_usd
                    else:
                        target_entry = target_obs.ask_usd if is_cheap else target_obs.bid_usd
                        atm_entry = atm_cand.bid_usd if is_cheap else atm_cand.ask_usd

                    if target_entry <= 0.0 or atm_entry <= 0.0:
                        continue

                    g_target = self.model.greeks(
                        smile.forward, target_obs.strike, smile.tau, target_obs.mid_iv, smile.rate, target_obs.option_type
                    )
                    vega_ratio = g_target.vega / g_atm.vega

                    # Calculate entry fees
                    side_mult = 1.0 if is_cheap else -1.0
                    atm_contracts = -side_mult * vega_ratio

                    inv_target = from_forward_greeks(target_entry, g_target, early_idx, smile.forward, smile.tau, smile.rate)
                    inv_atm = from_forward_greeks(atm_entry, g_atm, atm_cand.index_price, smile.forward, smile.tau, smile.rate)
                    portfolio_ntd = (side_mult * inv_target.net_transaction_delta) + (atm_contracts * inv_atm.net_transaction_delta)

                    if is_cheap:
                        e_buys = [(target_entry, 1.0)]
                        e_sells = [(atm_entry, abs(atm_contracts))]
                    else:
                        e_buys = [(atm_entry, abs(atm_contracts))]
                        e_sells = [(target_entry, 1.0)]

                    entry_opt_fee = calculate_option_combo_fee(e_buys, e_sells, early_idx, is_taker=True)
                    entry_hedge_fee = calculate_future_fee(smile.forward, abs(portfolio_ntd), is_taker=True)
                    est_roundtrip_fees = 2.0 * (entry_opt_fee + entry_hedge_fee)

                    theoretical_edge_usd = abs(price_res_loo)
                    if theoretical_edge_usd < (self.min_edge_fee_multiple * est_roundtrip_fees):
                        continue

                    open_positions[target_obs.instrument_name] = OpenPosition(
                        entry_snapshot_id=snap_curr.snapshot_id,
                        entry_timestamp_ns=snap_curr.timestamp_ns,
                        expiry=expiry,
                        target_instrument=target_obs.instrument_name,
                        target_strike=target_obs.strike,
                        target_type=target_obs.option_type,
                        atm_instrument=atm_cand.instrument_name,
                        side="BUY_DISLOCATED" if is_cheap else "SELL_DISLOCATED",
                        target_entry_usd=target_entry,
                        atm_entry_usd=atm_entry,
                        atm_contracts=atm_contracts,
                        portfolio_ntd=portfolio_ntd,
                        entry_forward=smile.forward,
                        initial_z_score=z_loo,
                        entry_fee_usd=entry_opt_fee + entry_hedge_fee,
                    )

        return closed_trades