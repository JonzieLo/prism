"""
SVI Dislocation Reversion Strategy Engine with Leave-One-Out (LOO) Fitting.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional
from deribit.chain import option_chain_from_snapshot
from deribit.pricing.black_scholes import Black76Model
from deribit.store import SnapshotStore
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.snapshot_loader import load_snapshot_observations


@dataclass
class TradePosition:
    entry_snapshot_id: int
    instrument_name: str
    expiry: int
    strike: float
    option_type: str
    side: str  # 'BUY_CHEAP' or 'SELL_RICH'
    entry_price_usd: float
    entry_forward: float
    entry_delta: float
    contracts: float


@dataclass
class ClosedTrade:
    entry_snapshot_id: int
    exit_snapshot_id: int
    instrument_name: str
    side: str
    entry_price_usd: float
    exit_price_usd: float
    option_pnl_usd: float
    hedge_pnl_usd: float
    fee_pnl_usd: float
    net_pnl_usd: float


class SVIReversionStrategy:
    def __init__(
        self,
        db_path: str = "snapshots.db",
        currency: str = "BTC",
        entry_threshold_bps: float = 15.0,
        taker_fee_bps: float = 3.0,
        max_abs_k: float = 0.75,
        use_leave_one_out: bool = True,
    ):
        self.store = SnapshotStore(db_path)
        self.currency = currency
        self.entry_threshold_bps = entry_threshold_bps
        self.taker_fee_bps = taker_fee_bps
        self.max_abs_k = max_abs_k
        self.use_leave_one_out = use_leave_one_out
        self.model = Black76Model()

    def run_backtest(self, snapshot_limit: int = 500) -> List[ClosedTrade]:
        snapshots = self.store.list_snapshots(currency=self.currency, limit=snapshot_limit)
        snapshots = sorted(snapshots, key=lambda s: s.timestamp_ns)

        if len(snapshots) < 2:
            return []

        closed_trades: List[ClosedTrade] = []

        for i in range(len(snapshots) - 1):
            snap_early = snapshots[i]
            snap_late = snapshots[i + 1]

            try:
                obs_early = load_snapshot_observations(self.store, snap_early.snapshot_id, max_abs_k=self.max_abs_k)
                snap_late_data = self.store.load_snapshot(snap_late.snapshot_id)
                late_chain = option_chain_from_snapshot(snap_late_data)
                late_quote_map = {q.instrument_name: q for q in late_chain.quotes}
            except Exception:
                continue

            for expiry, obs_list in obs_early.observations_by_expiry.items():
                if len(obs_list) < 6:
                    continue

                for target_obs in obs_list:
                    # Leave-One-Out (LOO) SVI Fitting
                    if self.use_leave_one_out:
                        fit_slice = [obs for obs in obs_list if obs.source_row_id != target_obs.source_row_id]
                    else:
                        fit_slice = obs_list

                    try:
                        smile_early = calibrate_svi_slice(fit_slice, require_butterfly_free=True)
                    except Exception:
                        continue

                    # Fair Value Calculation
                    k = target_obs.log_moneyness
                    tv_vol = float(smile_early.parameters.implied_vol(k, smile_early.tau))
                    tv_usd = self.model.price(
                        smile_early.forward, target_obs.strike, smile_early.tau, tv_vol, smile_early.rate, target_obs.option_type
                    )
                    
                    half_spread = (self.entry_threshold_bps / 10000.0) * smile_early.forward

                    # Match exit quote by instrument name
                    late_quote = late_quote_map.get(target_obs.instrument_name)
                    if not late_quote or late_quote.mid_usd is None:
                        continue

                    late_index = late_quote.index_price
                    late_bid_usd = (late_index * late_quote.bid_coin) if late_quote.bid_coin else None
                    late_ask_usd = (late_index * late_quote.ask_coin) if late_quote.ask_coin else None

                    # Signal 1: Buy Cheap Option
                    if target_obs.ask_usd < (tv_usd - half_spread) and target_obs.ask_usd > 0 and late_bid_usd:
                        entry_fee = target_obs.ask_usd * (self.taker_fee_bps / 10000.0)
                        exit_fee = late_bid_usd * (self.taker_fee_bps / 10000.0)
                        
                        entry_price = target_obs.ask_usd
                        exit_price = late_bid_usd

                        greeks = self.model.greeks(
                            smile_early.forward, target_obs.strike, smile_early.tau, tv_vol, smile_early.rate, target_obs.option_type
                        )
                        delta = greeks.delta
                        
                        # Executable forward change for delta hedge
                        late_forward = late_quote.index_price  # Approximate forward
                        spot_change = late_forward - smile_early.forward
                        hedge_pnl = -delta * spot_change

                        opt_pnl = exit_price - entry_price
                        total_fees = entry_fee + exit_fee
                        net_pnl = opt_pnl + hedge_pnl - total_fees

                        closed_trades.append(
                            ClosedTrade(
                                entry_snapshot_id=snap_early.snapshot_id,
                                exit_snapshot_id=snap_late.snapshot_id,
                                instrument_name=target_obs.instrument_name,
                                side="BUY_CHEAP",
                                entry_price_usd=entry_price,
                                exit_price_usd=exit_price,
                                option_pnl_usd=opt_pnl,
                                hedge_pnl_usd=hedge_pnl,
                                fee_pnl_usd=-total_fees,
                                net_pnl_usd=net_pnl,
                            )
                        )

                    # Signal 2: Sell Rich Option
                    elif target_obs.bid_usd > (tv_usd + half_spread) and late_ask_usd:
                        entry_fee = target_obs.bid_usd * (self.taker_fee_bps / 10000.0)
                        exit_fee = late_ask_usd * (self.taker_fee_bps / 10000.0)
                        
                        entry_price = target_obs.bid_usd
                        exit_price = late_ask_usd

                        greeks = self.model.greeks(
                            smile_early.forward, target_obs.strike, smile_early.tau, tv_vol, smile_early.rate, target_obs.option_type
                        )
                        delta = greeks.delta
                        
                        late_forward = late_quote.index_price
                        spot_change = late_forward - smile_early.forward
                        hedge_pnl = delta * spot_change

                        opt_pnl = entry_price - exit_price
                        total_fees = entry_fee + exit_fee
                        net_pnl = opt_pnl + hedge_pnl - total_fees

                        closed_trades.append(
                            ClosedTrade(
                                entry_snapshot_id=snap_early.snapshot_id,
                                exit_snapshot_id=snap_late.snapshot_id,
                                instrument_name=target_obs.instrument_name,
                                side="SELL_RICH",
                                entry_price_usd=entry_price,
                                exit_price_usd=exit_price,
                                option_pnl_usd=opt_pnl,
                                hedge_pnl_usd=hedge_pnl,
                                fee_pnl_usd=-total_fees,
                                net_pnl_usd=net_pnl,
                            )
                        )

        return closed_trades