from dataclasses import dataclass
import math
from typing import Dict, List, Optional
from deribit.pricing.black_scholes import Black76Model
from deribit.surface.results import CalibratedSmile


@dataclass(frozen=True)
class QuoteConfig:
    min_half_width_usd: float = 4.30
    target_half_vol_points: fload = 1.5
    rmse_weight: float = 1.0
    vega_decay_weight: float = 0.05
    market_spread_weight: float = 0.10
    inventory_gamma: float = 0.002
    max_inventory_vega: float = 50.0


@dataclass(frozen=True)
class SimulatedQuote:
    instrument_name: str
    strike: float
    tau: float
    option_type: str
    tv_usd: float
    half_width_usd: float
    shaded_bid_usd: float
    shaded_ask_usd: float
    market_bid_usd: float
    market_ask_usd: float
    bid_relation: str  # 'INSIDE', 'OUTSIDE', or 'IMPROVED'
    ask_relation: str
    is_crossed: bool


@dataclass(frozen=True)
class FillEvent:
    timestamp_ns: int
    instrument_name: str
    side: str  # 'BUY' or 'SELL'
    fill_price_usd: float
    tv_at_fill_usd: float
    edge_captured_usd: float
    contracts: float
    vega: float


class QuotingSimulator:
    def __init__(self, config: QuoteConfig = QuoteConfig()):
        self.config = config
        self.inventory_vega: float = 0.0
        self.inventory_contracts: Dict[str, float] = {}
        self.fills: List[FillEvent] = []
        self.model = Black76Model()

    def derive_quote_width(
        self,
        forward: float,
        tau: float,
        vega: float,
        tv_vol: float,
        rmse_total_variance: float,
        market_bid_usd: float,
        market_ask_usd: float,
    ) -> float:
        """
        Derives quote half-width in USD:
        w_half = BaseWidth + (RMSE_vol * Vega) + VolDecayPenalty + MarketSpreadFraction
        """
        vol_width_usd = vega * (self.config.target_half_vol_points / 100.0)
        base_width = max(self.config.min_half_width_usd, vol_width_usd)

        vol_rmse = rmse_total_variance / (2.0 * max(tau, 1.0/365.0) * max(tv_vol, 0.1))
        fit_penalty = self.config.rmse_weight * vega * vol_rmse
        
        market_spread = max(0.0, market_ask_usd - market_bid_usd)
        mkt_penalty = self.config.market_spread_weight * market_spread
        
        return base_width + fit_penalty + mkt_penalty

    def generate_quote(
        self,
        smile: CalibratedSmile,
        strike: float,
        option_type: str,
        instrument_name: str,
        market_bid_usd: float,
        market_ask_usd: float,
    ) -> Optional[SimulatedQuote]:
        if strike <= 0.0 or smile.forward <= 0.0 or smile.tau <= 0.0:
            return None

        if abs(self.inventory_vega) >= self.config.max_inventory_vega:
            return None

        k = math.log(strike / smile.forward)
        tv_vol = float(smile.parameters.implied_vol(k, smile.tau))

        tv_usd = self.model.price(smile.forward, strike, smile.tau, tv_vol, smile.rate, option_type)
        greeks = self.model.greeks(smile.forward, strike, smile.tau, tv_vol, smile.rate, option_type)
        vega = greeks.vega

        half_width = self.derive_quote_width(
            smile.forward, smile.tau, vega, tv_vol, smile.rmse_total_variance, market_bid_usd, market_ask_usd
        )

        # Inventory shading: skew down if long vega, shade up if short vega
        inventory_skew = self.config.inventory_gamma * self.inventory_vega * vega
        shaded_mid = tv_usd - inventory_skew
        shaded_bid = max(0.0, shaded_mid - half_width)
        shaded_ask = max(shaded_bid + 0.01, shaded_mid + half_width)

        bid_rel = "IMPROVED" if shaded_bid > market_bid_usd else ("INSIDE" if shaded_bid == market_bid_usd else "OUTSIDE")
        ask_rel = "IMPROVED" if shaded_ask < market_ask_usd else ("INSIDE" if shaded_ask == market_ask_usd else "OUTSIDE")
        
        # Check if quote is marketable / crossed against external book
        is_crossed = (shaded_bid >= market_ask_usd > 0.0) or (shaded_ask <= market_bid_usd > 0.0)

        return SimulatedQuote(
            instrument_name=instrument_name,
            strike=strike,
            tau=smile.tau,
            option_type=option_type,
            tv_usd=tv_usd,
            half_width_usd=half_width,
            shaded_bid_usd=shaded_bid,
            shaded_ask_usd=shaded_ask,
            market_bid_usd=market_bid_usd,
            market_ask_usd=market_ask_usd,
            bid_relation=bid_rel,
            ask_relation=ask_rel,
            is_crossed=is_crossed,
        )

    def process_fill(
        self,
        quote: SimulatedQuote,
        side: str,
        fill_price_usd: float,
        timestamp_ns: int,
        vega: float,
        contracts: float = 1.0,
    ) -> FillEvent:
        edge = (quote.tv_usd - fill_price_usd) if side == "BUY" else (fill_price_usd - quote.tv_usd)
        
        fill = FillEvent(
            timestamp_ns=timestamp_ns,
            instrument_name=quote.instrument_name,
            side=side,
            fill_price_usd=fill_price_usd,
            tv_at_fill_usd=quote.tv_usd,
            edge_captured_usd=edge * contracts,
            contracts=contracts,
            vega=vega,
        )
        
        self.fills.append(fill)

        direction = 1.0 if side == "BUY" else -1.0
        self.inventory_vega += direction * vega * contracts
        self.inventory_contracts[quote.instrument_name] = (
            self.inventory_contracts.get(quote.instrument_name, 0.0) + (direction * contracts)
        )
        
        return fill


if __name__ == "__main__":
    import argparse
    from deribit.store import SnapshotStore, fetch_and_save_snapshot
    from deribit.surface.calibration import calibrate_svi_slice
    from deribit.surface.snapshot_loader import load_snapshot_observations

    parser = argparse.ArgumentParser(description="Run M7 Live Quoting Simulator.")
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--snapshot-id", type=int)
    parser.add_argument("--fetch", action="store_true", help="Fetch a fresh live snapshot before quoting")
    parser.add_argument("--prod", action="store_true", help="Use Deribit Production (mainnet) for live fetch")
    args = parser.parse_args()

    store = SnapshotStore(args.db)

    if args.fetch:
        print(f"Fetching live snapshot from Deribit ({'Prod' if args.prod else 'Testnet'})...")
        snap_id = fetch_and_save_snapshot(store, currency=args.currency, testnet=not args.prod)
    else:
        snap_id = args.snapshot_id or store.latest_snapshot_id(args.currency)

    if not snap_id:
        print("No snapshots found. Run with --fetch to capture one.")
        exit(1)

    loaded = load_snapshot_observations(store, snap_id)
    simulator = QuotingSimulator(config=QuoteConfig(inventory_gamma=0.002))

    print("=" * 80)
    print(f"=== LIVE QUOTING SIMULATOR (Snapshot #{snap_id}) ===")
    print("=" * 80)

    for expiry, obs_list in loaded.observations_by_expiry.items():
        smile = calibrate_svi_slice(obs_list)

        for obs in obs_list[:3]:
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

            print(f"\nInstrument:   {quote.instrument_name}")
            print(f"  Fair Value: ${quote.tv_usd:,.2f} | Half-Width: ${quote.half_width_usd:,.2f}")
            print(f"  Market:     ${quote.market_bid_usd:,.2f} / ${quote.market_ask_usd:,.2f}")
            print(f"  Shaded:     ${quote.shaded_bid_usd:,.2f} / ${quote.shaded_ask_usd:,.2f}")
            print(f"  Bid Rel:    {quote.bid_relation:<8} | Ask Rel: {quote.ask_relation:<8} | Crossed: {quote.is_crossed}")