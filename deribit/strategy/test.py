"""
PRISM Quantitative Options Strategy: GARCH-Filtered Volatility Harvesting & Validation Plotter.

Combines:
1. Deribit API Historical Data Fetching (DVOL + Spot).
2. GJR-GARCH 30-Day Forward Volatility Forecasting.
3. Black-76 Exact Delta-Hedged Lifecycle Backtest (7-day holding).
4. Automated 2-Panel Validation Figure Generation (vrp_harvest_validation.png).
"""

from datetime import datetime, timezone
import time
import httpx
import matplotlib.pyplot as plt
import numpy as np

from deribit.pricing.black_scholes import Black76Model
from deribit.strategy.garch import GJR_GARCH


def to_utc_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def fetch_deribit_history(days: int = 400):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 86400 * 1000)

    print(f"Fetching {days} days of historical market data from Deribit...")

    # 1. Fetch DVOL
    dvol_url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    dvol_params = {
        "currency": "BTC",
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "resolution": "1D",
    }
    dvol_data = httpx.get(dvol_url, params=dvol_params, timeout=15.0).json()["result"]["data"]

    dvol_by_date = {}
    for row in dvol_data:
        val = float(row[4])
        dvol_by_date[to_utc_date(row[0])] = val / 100.0 if val > 2.0 else val

    # 2. Fetch Spot Closes
    spot_url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    spot_params = {
        "instrument_name": "BTC-PERPETUAL",
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "resolution": "1D",
    }
    spot_res = httpx.get(spot_url, params=spot_params, timeout=15.0).json()["result"]
    spot_ticks = spot_res["ticks"]
    spot_closes = spot_res["close"]

    records = []
    for t, price in zip(spot_ticks, spot_closes):
        d_str = to_utc_date(t)
        if d_str in dvol_by_date:
            records.append((d_str, float(price), dvol_by_date[d_str]))

    print(f"-> Successfully aligned {len(records)} daily observations.\n")
    return records


def plot_vrp_validation(records, burn_in, garch_history, trades, output_path="vrp_harvest_validation.png"):
    n_pts = len(garch_history)
    eval_records = records[burn_in : burn_in + n_pts]
    eval_dates = [r[0] for r in eval_records]
    eval_spots = [r[1] for r in eval_records]
    eval_dvol = [r[2] * 100.0 for r in eval_records]
    eval_garch = [g * 100.0 for g in garch_history]

    trade_exit_map = {t["exit_date"]: t["net_pnl"] for t in trades}
    pnl_series = []
    running_pnl = 0.0

    for d in eval_dates:
        if d in trade_exit_map:
            running_pnl += trade_exit_map[d]
        pnl_series.append(running_pnl)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")

    # --- Panel 1: Strategy P&L vs Spot ---
    line_pnl = ax1.plot(eval_dates, pnl_series, color="#1a1a1a", lw=1.8, label="Cumulative P&L ($)")
    ax1.set_ylabel("Net P&L (USD)")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_spot = ax1.twinx()
    line_spot = ax1_spot.plot(eval_dates, eval_spots, color="#777777", lw=1.2, ls="--", label="BTC Spot ($)")
    ax1_spot.set_ylabel("BTC Spot (USD)")

    # Combine legends cleanly without warnings
    lines = line_pnl + line_spot
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)
    ax1.set_title("VRP Harvesting Backtest (30d Straddle, 7d Hold)")

    # --- Panel 2: Implied Vol vs GARCH Forecast ---
    ax2.plot(eval_dates, eval_dvol, color="#1f77b4", lw=1.4, label="Deribit DVOL (30d IV)")
    ax2.plot(eval_dates, eval_garch, color="#d62728", lw=1.4, ls="--", label="GJR-GARCH Forecast (30d)")

    # Subtle gray shading for active trades
    for idx, t in enumerate(trades):
        lbl = "Active Trade" if idx == 0 else None
        ax2.axvspan(t["entry_date"], t["exit_date"], color="#dddddd", alpha=0.5, label=lbl)

    ax2.set_ylabel("Volatility (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    # Subsample date labels along x-axis
    tick_step = max(1, len(eval_dates) // 8)
    ax2.set_xticks(eval_dates[::tick_step])
    ax2.tick_params(axis="x", rotation=20)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"\nSaved validation chart to: {output_path}")


def run_lifecycle_backtest(
    days: int = 400,
    hold_days: int = 7,
    min_vrp_vol_points: float = 5.0,
    spread_haircut_vols: float = 0.02,
):
    records = fetch_deribit_history(days=days)
    if len(records) < 180:
        print("Need at least 180 days of history.")
        return

    dates = [r[0] for r in records]
    prices = np.array([r[1] for r in records])
    ivs = np.array([r[2] for r in records])
    log_returns = np.diff(np.log(prices))

    garch = GJR_GARCH()
    b76 = Black76Model()

    # Calibrate GARCH on initial burn-in
    burn_in = 150
    initial_returns = log_returns[:burn_in]
    garch_params = garch.fit(initial_returns)
    variance_path = garch.filter_variance(initial_returns, garch_params)
    current_var = variance_path[-1]

    print("=" * 80)
    print("=== LIFECYCLE DELTA-HEDGED STRADDLE BACKTEST ===")
    print(f"GARCH Fit:          omega={garch_params.omega:.2e}, alpha={garch_params.alpha:.3f}, beta={garch_params.beta:.3f}")
    print(f"GARCH Long-Run Vol: {garch_params.unconditional_vol_annualized * 100:.2f}%")
    print(f"Holding Period:     {hold_days} days per trade")
    print(f"Spread Haircut:     {spread_haircut_vols * 100:.1f} vol points (paid once per cycle)")
    print("=" * 80)

    trades = []
    garch_forecast_history = []
    i = burn_in
    rate = 0.03

    while i < len(records) - 1:
        s_0 = prices[i]
        iv_0 = ivs[i]

        shock = log_returns[i - 1] - garch_params.mu
        leverage = garch_params.gamma * (shock ** 2) if shock < 0 else 0.0
        current_var = (
            garch_params.omega
            + garch_params.alpha * (shock ** 2)
            + leverage
            + garch_params.beta * current_var
        )

        garch_rv = garch.forecast_realized_vol(garch_params, current_var, shock, horizon_days=30.0)
        garch_forecast_history.append(garch_rv)

        vrp = (iv_0 - garch_rv) * 100.0

        # Check VRP entry condition & ensure enough remaining days to hold
        if vrp < min_vrp_vol_points or (i + hold_days >= len(records)):
            i += 1
            continue

        # OPEN 30-DAY STRADDLE
        tau_0 = 30.0 / 365.0
        exec_entry_iv = max(0.10, iv_0 - spread_haircut_vols)
        forward_0 = s_0 * np.exp(rate * tau_0)
        strike_atm = s_0

        entry_call = b76.price(forward_0, strike_atm, tau_0, exec_entry_iv, rate, "call")
        entry_put = b76.price(forward_0, strike_atm, tau_0, exec_entry_iv, rate, "put")
        collected_premium = entry_call + entry_put

        cycle_hedge_pnl = 0.0
        curr_forward = forward_0
        curr_tau = tau_0
        curr_iv = exec_entry_iv

        for day in range(1, hold_days + 1):
            s_d = prices[i + day]
            curr_tau = (30.0 - day) / 365.0
            next_forward = s_d * np.exp(rate * curr_tau)

            c_g = b76.greeks(curr_forward, strike_atm, curr_tau + 1.0/365.0, curr_iv, rate, "call")
            p_g = b76.greeks(curr_forward, strike_atm, curr_tau + 1.0/365.0, curr_iv, rate, "put")
            net_delta = c_g.delta + p_g.delta

            cycle_hedge_pnl += net_delta * (next_forward - curr_forward)
            curr_forward = next_forward
            curr_iv = ivs[i + day]

            # Track intermediate GARCH forecasts during the hold
            if day < hold_days:
                s_int = log_returns[i + day - 1] - garch_params.mu
                lev_int = garch_params.gamma * (s_int ** 2) if s_int < 0 else 0.0
                current_var = (
                    garch_params.omega
                    + garch_params.alpha * (s_int ** 2)
                    + lev_int
                    + garch_params.beta * current_var
                )
                g_rv_int = garch.forecast_realized_vol(garch_params, current_var, s_int, horizon_days=30.0)
                garch_forecast_history.append(g_rv_int)

        # CLOSE POSITION ON DAY 7
        s_exit = prices[i + hold_days]
        iv_exit = ivs[i + hold_days]
        tau_exit = (30.0 - hold_days) / 365.0
        forward_exit = s_exit * np.exp(rate * tau_exit)

        exec_exit_iv = iv_exit + spread_haircut_vols
        exit_call = b76.price(forward_exit, strike_atm, tau_exit, exec_exit_iv, rate, "call")
        exit_put = b76.price(forward_exit, strike_atm, tau_exit, exec_exit_iv, rate, "put")
        repurchase_cost = exit_call + exit_put

        option_pnl = collected_premium - repurchase_cost
        net_trade_pnl = option_pnl + cycle_hedge_pnl

        trades.append({
            "entry_date": dates[i],
            "exit_date": dates[i + hold_days],
            "vrp": vrp,
            "option_pnl": option_pnl,
            "hedge_pnl": cycle_hedge_pnl,
            "net_pnl": net_trade_pnl,
        })

        i += hold_days

    net_pnls = np.array([t["net_pnl"] for t in trades])
    win_rate = np.mean(net_pnls > 0)
    total_pnl = np.sum(net_pnls)
    mean_pnl = np.mean(net_pnls)
    std_pnl = np.std(net_pnls)
    annualized_sharpe = (mean_pnl / std_pnl) * np.sqrt(365.0 / hold_days) if std_pnl > 0 else 0.0

    print(f"Total Cycles Traded:      {len(trades)}")
    print(f"Win Rate:                 {win_rate:.2%}")
    print(f"Cumulative Net P&L:       ${total_pnl:,.2f} per 1 BTC straddle")
    print(f"Mean P&L per 7-Day Cycle: ${mean_pnl:,.2f}")
    print(f"Cycle P&L StdDev:         ${std_pnl:,.2f}")
    print(f"Annualized Sharpe Ratio:  {annualized_sharpe:.2f}")

    # Generate Chart
    plot_vrp_validation(records, burn_in, garch_forecast_history, trades)


if __name__ == "__main__":
    run_lifecycle_backtest(days=400, hold_days=7, min_vrp_vol_points=5.0)