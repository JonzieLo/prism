from dataclasses import dataclass
import math
from typing import Dict, List, Optional
from deribit.pricing.black_scholes import Black76Model
from deribit.surface.results import CalibratedSmile


@dataclass(frozen=True)
class QuoteConfig:
    min_half_width_usd: float = 4.30      # Deribit minimum half-tick
    target_half_vol_points: float = 1.5   # Quote ~1.5 vol points wide
    rmse_weight: float = 0.10
    market_spread_weight: float = 0.10
    inventory_gamma: float = 0.05         # Vol-point skew per unit of 1-vol vega
    max_inventory_vega: float = 500.0     # Max inventory in 1-vol vega units (~$500/vol)


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
    bid_relation: str
    ask_relation: str
    is_crossed: bool


@dataclass(frozen=True)
class FillEvent:
    timestamp_ns: int
    instrument_name: str
    side: str
    fill_price_usd: float
    tv_at_fill_usd: float
    edge_captured_usd: float
    contracts: float
    vega_1vol: float


class QuotingSimulator:
    def __init__(self, config: QuoteConfig = QuoteConfig()):
        self.config = config
        self.inventory_vega: float = 0.0  # Tracked in 1-vol vega ($/1% vol)
        self.inventory_contracts: Dict[str, float] = {}
        self.fills: List[FillEvent] = []
        self.model = Black76Model()

    def derive_quote_width(
        self,
        forward: float,
        tau: float,
        vega_1vol: float,
        tv_vol: float,
        rmse_total_variance: float,
        market_bid_usd: float,
        market_ask_usd: float,
    ) -> float:
        # Base width = 1.5 vol points * vega_1vol, floored at min tick ($4.30)
        vol_width_usd = vega_1vol * self.config.target_half_vol_points
        base_width = max(self.config.min_half_width_usd, vol_width_usd)

        clamped_tau = max(tau, 1.0 / 365.0)
        vol_rmse = min(0.20, rmse_total_variance / (2.0 * clamped_tau * max(tv_vol, 0.10)))
        fit_penalty = self.config.rmse_weight * vega_1vol * vol_rmse * 100.0

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
        
        # Normalize Vega to 1-vol points ($/1% vol move)
        vega_1vol = greeks.vega / 100.0

        half_width = self.derive_quote_width(
            smile.forward, smile.tau, vega_1vol, tv_vol, smile.rmse_total_variance, market_bid_usd, market_ask_usd
        )

        # Inventory shading: skew price by at most 30% of fair value or 1.5x half_width
        raw_skew = self.config.inventory_gamma * (self.inventory_vega / 100.0) * vega_1vol
        max_allowed_skew = min(0.30 * tv_usd, 1.5 * half_width)
        inventory_skew = max(-max_allowed_skew, min(max_allowed_skew, raw_skew))

        shaded_mid = tv_usd - inventory_skew
        raw_bid = max(0.01, shaded_mid - half_width)
        raw_ask = max(raw_bid + 0.01, shaded_mid + half_width)

        # Ensure market maker quotes never cross through the opposite side of the book
        shaded_bid = min(raw_bid, market_ask_usd - 0.50) if market_ask_usd > 1.0 else raw_bid
        shaded_ask = max(raw_ask, market_bid_usd + 0.50) if market_bid_usd > 0.0 else raw_ask

        bid_rel = "IMPROVED" if shaded_bid > market_bid_usd else ("INSIDE" if shaded_bid == market_bid_usd else "OUTSIDE")
        ask_rel = "IMPROVED" if shaded_ask < market_ask_usd else ("INSIDE" if shaded_ask == market_ask_usd else "OUTSIDE")
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
        vega_1vol: float,
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
            vega_1vol=vega_1vol,
        )
        self.fills.append(fill)

        direction = 1.0 if side == "BUY" else -1.0
        self.inventory_vega += direction * vega_1vol * contracts
        self.inventory_contracts[quote.instrument_name] = (
            self.inventory_contracts.get(quote.instrument_name, 0.0) + (direction * contracts)
        )
        return fill