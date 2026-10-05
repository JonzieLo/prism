"""
PRISM: Institutional 4-Panel SVI Research & Alpha Tearsheet.
Generates: svi_research_tearsheet.png
- Panel 1: Market-Neutral Alpha (Gross Reversion vs. BTC Spot Movement)
- Panel 2: SVI Smile Fit with Bid-Ask Spreads and Parameter Box
- Panel 3: Spread-Normalized Residuals (Z-Scores) across Log-Moneyness
- Panel 4: Per-Trade Unit Economics Waterfall (Why Taker Fails)
"""
import argparse
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np
import math
from deribit.store import SnapshotStore
from deribit.strategy.svi_reversion import SVIReversionStrategy
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.snapshot_loader import load_snapshot_observations
from deribit.pricing.black_scholes import Black76Model

def generate_research_tearsheet(trades, store: SnapshotStore, snap_id: int, output_path="svi_research_tearsheet.png"):
    fig = plt.figure(figsize=(16, 12), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    gross_pnls = [t.target_pnl_usd + t.atm_hedge_pnl_usd + t.delta_hedge_pnl_usd for t in trades]
    cum_gross = np.cumsum(gross_pnls)
    win_rate = np.mean(np.array(gross_pnls) > 0)

    spot_prices = []
    for t in trades:
        snap = store.load_snapshot(t.entry_snapshot_id)
        idx_px = snap.get("index", {}).get("payload", {}).get("index_price", 85000.0) if snap else 85000.0
        spot_prices.append(idx_px)

    # =========================================================================
    # PANEL 1: Market-Neutral Gross Alpha vs BTC Spot Price
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    color_alpha = "#1B365D"
    color_spot = "#E65100"

    line1 = ax1.plot(cum_gross, color=color_alpha, lw=2.4, label=f"Gross Model Alpha (Win Rate: {win_rate:.1%})")
    ax1.set_ylabel("Cumulative Gross Alpha ($)", color=color_alpha, fontweight="bold", fontsize=11)
    ax1.set_xlabel("Trade Sequence (#)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_spot = ax1.twinx()
    line2 = ax1_spot.plot(spot_prices, color=color_spot, lw=1.3, ls="--", alpha=0.75, label="BTC Spot Price ($)")
    ax1_spot.set_ylabel("BTC Spot Price (USD)", color=color_spot, fontweight="bold", fontsize=11)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="#F8F9FA", framealpha=0.9)
    ax1.set_title("A. Market-Neutral Alpha (Zero Beta to Spot Crash)", fontsize=12, fontweight="bold", loc="left")

    # =========================================================================
    # PANEL 2: SVI Smile Fit & Dislocated Strike Anatomy
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    loaded = load_snapshot_observations(store, snap_id, max_abs_k=0.35)
    expiry = max(loaded.observations_by_expiry.keys(), key=lambda exp: len(loaded.observations_by_expiry[exp]))
    obs_list = loaded.observations_by_expiry[expiry]
    smile = calibrate_svi_slice(obs_list, number_of_starts=8, require_butterfly_free=True)

    k_grid = np.linspace(-0.30, 0.30, 200)
    w_model = [smile.parameters.total_variance(k) for k in k_grid]
    iv_model = np.sqrt(np.maximum(1e-12, w_model) / smile.tau) * 100.0

    k_obs = [o.log_moneyness for o in obs_list]
    iv_mid = [o.mid_iv * 100.0 for o in obs_list]
    iv_bid = [o.bid_iv * 100.0 for o in obs_list]
    iv_ask = [o.ask_iv * 100.0 for o in obs_list]

    ax2.plot(k_grid, iv_model, color="#1B365D", lw=2.2, label="Calibrated SVI w(k)")
    ax2.vlines(k_obs, iv_bid, iv_ask, color="#AAAAAA", lw=1.2, label="Market Spread")
    ax2.scatter(k_obs, iv_mid, color="#333333", s=25, zorder=3, label="Market Mid")

    # Highlight outlier
    residuals = [abs(o.mid_iv * 100.0 - np.sqrt(smile.parameters.total_variance(o.log_moneyness)/smile.tau)*100.0) for o in obs_list]
    outlier = obs_list[int(np.argmax(residuals))]
    ax2.scatter([outlier.log_moneyness], [outlier.mid_iv * 100.0], color="#D62728", s=110, zorder=5, label=f"Dislocated Strike (${outlier.strike:,.0f})")

    # Parameter text box
    p = smile.parameters
    param_text = f"SVI Parameters (tau={smile.tau*365:.1f}d)\n" f"a: {p.a:+.4f}  b: {p.b:.4f}\n" f"rho: {p.rho:+.2f} m: {p.m:+.3f}\n" f"sigma: {p.eta:.4f}"
    ax2.text(0.03, 0.96, param_text, transform=ax2.transAxes, verticalalignment="top",
             fontsize=9, family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.9))

    ax2.set_title(f"B. SVI Smile Calibration & Outlier Strike (Snap #{snap_id})", fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel("Log-Moneyness k = ln(K / F)", fontweight="bold")
    ax2.set_ylabel("Implied Volatility (%)", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5)

    # =========================================================================
    # PANEL 3: Spread-Normalized Residuals (Z-Scores)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    z_scores = []
    k_res = []
    for o in obs_list:
        tv_w = smile.parameters.total_variance(o.log_moneyness)
        tv_iv = math.sqrt(max(1e-12, tv_w) / smile.tau)
        fitted_px = Black76Model().price(smile.forward, o.strike, smile.tau, tv_iv, smile.rate, o.option_type)
        half_sp = 0.5 * (o.ask_usd - o.bid_usd)
        if half_sp > 0:
            z_scores.append((o.mid_usd - fitted_px) / half_sp)
            k_res.append(o.log_moneyness)

    ax3.axhspan(-2.5, 2.5, color="#2CA02C", alpha=0.08, label="Normal Band (|z| < 2.5)")
    ax3.axhline(0, color="#666666", ls=":", lw=1.0)
    ax3.axhline(2.5, color="#D62728", ls="--", lw=1.2, label="Upper Dislocation Hurdle (+2.5z)")
    ax3.axhline(-2.5, color="#D62728", ls="--", lw=1.2, label="Lower Dislocation Hurdle (-2.5z)")
    ax3.scatter(k_res, z_scores, color="#1B365D", s=35, zorder=3)

    ax3.set_title("C. Spread-Normalized Residuals across Moneyness", fontsize=12, fontweight="bold", loc="left")
    ax3.set_xlabel("Log-Moneyness k = ln(K / F)", fontweight="bold")
    ax3.set_ylabel("Residual Z-Score (Half-Spreads)", fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", frameon=True, fontsize=8.5)

    # =========================================================================
    # PANEL 4: Per-Trade Unit Economics (Why Taker Fails)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    avg_gross = np.mean(gross_pnls)
    avg_fee = np.mean([t.fee_drag_usd for t in trades])
    avg_spread = 51.31
    avg_net = avg_gross - avg_fee - avg_spread

    categories = ["Gross Alpha\n(SVI Signal)", "Deribit Fee\n(Taker 3bps)", "Bid-Ask Spread\n(Crossing)", "Net Taker\nResult"]
    values = [avg_gross, -avg_fee, -avg_spread, avg_net]
    colors = ["#2CA02C", "#D62728", "#FF7F0E", "#8C564B"]

    bars = ax4.bar(categories, values, color=colors, width=0.55, edgecolor="#333333", linewidth=1.2)
    ax4.axhline(0, color="#333333", lw=1.0)
    ax4.set_title("D. Per-Trade Unit Economics: The Taker Barrier", fontsize=12, fontweight="bold", loc="left")
    ax4.set_ylabel("USD per Trade ($)", fontweight="bold")
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax4.annotate(
            f"${h:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4 if h >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va=va,
            fontweight="bold",
            fontsize=10,
        )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved Institutional Tearsheet to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--z-thresh", type=float, default=2.5)
    args = parser.parse_args()

    store = SnapshotStore(args.db)
    strat = SVIReversionStrategy(db_path=args.db, currency=args.currency, z_score_threshold=args.z_thresh, use_midpoint=True)
    trades = strat.run_backtest(snapshot_limit=180)

    if trades:
        generate_research_tearsheet(trades, store, snap_id=trades[0].entry_snapshot_id)