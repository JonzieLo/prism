"""
PRISM Quantitative Options Strategy: Continuous Volatility Harvesting Pipeline.

Upgrades from test_2.py:
1. Continuous 1/7th daily rolling tranches (up to 7 overlapping 7-day holds).
2. DVOL-to-ATM IV conversion (-3.0 vol points skew adjustment).
3. Rolling 180-day GJR-GARCH model refitting.
4. Futures rehedging frictions (3 bps taker fee on delta changes).
5. Continuous equity curve & multi-tranche validation plotting.
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


class OptionTranche:
    """Represents 1/7th capacity short straddle tranche held for up to 7 days."""

    def __init__(
        self,
        entry_idx: int,
        entry_date: str,
        spot_0: float,
        atm_iv_0: float,
        hold_days: int = 7,
        capacity_weight: float = 1.0 / 7.0,
        rate: float = 0.03,
        spread_haircut_vols: float = 0.02,
    ):
        self.entry_idx = entry_idx
        self.entry_date = entry_date
        self.spot_0 = spot_0
        self.strike = spot_0  # ATM straddle
        self.hold_days = hold_days
        self.weight = capacity_weight
        self.rate = rate
        self.spread_haircut = spread_haircut_vols

        b76 = Black76Model()
        tau_0 = 30.0 / 365.0
        forward_0 = spot_0 * np.exp(rate * tau_0)
        exec_entry_iv = max(0.10, atm_iv_0 - spread_haircut_vols)

        c0 = b76.price(forward_0, self.strike, tau_0, exec_entry_iv, rate, "call")
        p0 = b76.price(forward_0, self.strike, tau_0, exec_entry_iv, rate, "put")

        self.entry_iv = exec_entry_iv
        self.collected_premium = (c0 + p0) * self.weight
        self.prev_delta = 0.0
        self.days_held = 0
        self.is_closed = False


def run_continuous_vrp_backtest(
    days: int = 400,
    hold_days: int = 7,
    min_vrp_vol_points: float = 5.0,
    spread_haircut_vols: float = 0.02,
    atm_skew_discount_vols: float = 0.03,  # Subtract 3 vol pts from DVOL for ATM IV
    perp_fee_bps: float = 3.0,            # 3 bps taker fee on futures rehedges
):
    records = fetch_deribit_history(days=days)
    if len(records) < 180:
        print("Need at least 180 days of history.")
        return

    dates = [r[0] for r in records]
    prices = np.array([r[1] for r in records])
    dvol_ivs = np.array([r[2] for r in records])
    log_returns = np.diff(np.log(prices))

    b76 = Black76Model()
    garch = GJR_GARCH()

    burn_in = 150
    rate = 0.03
    perp_fee_rate = perp_fee_bps / 10000.0

    # Initial GARCH Calibration
    garch_params = garch.fit(log_returns[:burn_in])
    variance_path = garch.filter_variance(log_returns[:burn_in], garch_params)
    current_var = variance_path[-1]

    print("=" * 80)
    print("=== CONTINUOUS TRANCHE DELTA-HEDGED VRP BACKTEST ===")
    print(f"Capacity Split:      7 Daily Overlapping Tranches (1/7th allocation each)")
    print(f"ATM IV Adjustment:   DVOL - {atm_skew_discount_vols * 100:.1f} vol pts")
    print(f"Rehedge Taker Fee:   {perp_fee_bps:.1f} bps per delta adjustment")
    print("=" * 80)

    active_tranches = []
    daily_pnl_history = []
    garch_forecasts = []
    portfolio_delta_history = []

    cum_pnl = 0.0

    for i in range(burn_in, len(records) - 1):
        s_curr = prices[i]
        dvol_curr = dvol_ivs[i]

        # 1. Periodic Rolling GARCH Refit (every 7 days)
        if (i - burn_in) % 7 == 0 and i > burn_in:
            lookback_returns = log_returns[max(0, i - 180) : i]
            try:
                garch_params = garch.fit(lookback_returns)
            except Exception:
                pass  # Fallback to existing parameters if optimization fails

        # 2. Update GARCH Variance State
        shock = log_returns[i - 1] - garch_params.mu
        leverage = garch_params.gamma * (shock ** 2) if shock < 0 else 0.0
        current_var = (
            garch_params.omega
            + garch_params.alpha * (shock ** 2)
            + leverage
            + garch_params.beta * current_var
        )

        garch_rv = garch.forecast_realized_vol(garch_params, current_var, shock, horizon_days=30.0)
        garch_forecasts.append(garch_rv)

        # 3. Estimate ATM IV (DVOL minus skew discount)
        atm_iv_curr = max(0.10, dvol_curr - atm_skew_discount_vols)
        vrp = (atm_iv_curr - garch_rv) * 100.0

        # 4. Open New 1/7th Tranche if VRP threshold is met
        if vrp >= min_vrp_vol_points:
            new_tranche = OptionTranche(
                entry_idx=i,
                entry_date=dates[i],
                spot_0=s_curr,
                atm_iv_0=atm_iv_curr,
                hold_days=hold_days,
                capacity_weight=1.0 / hold_days,
                rate=rate,
                spread_haircut_vols=spread_haircut_vols,
            )
            active_tranches.append(new_tranche)

        # 5. Process Active Tranches & Execute Daily Delta Rehedging
        daily_option_pnl = 0.0
        daily_hedge_pnl = 0.0
        daily_friction_cost = 0.0
        total_port_delta = 0.0

        next_spot = prices[i + 1]

        for tranche in list(active_tranches):
            tranche.days_held += 1
            tau_curr = (30.0 - tranche.days_held) / 365.0
            forward_curr = s_curr * np.exp(rate * tau_curr)
            forward_next = next_spot * np.exp(rate * (tau_curr - 1.0 / 365.0))

            # Calculate Greeks for Short Straddle (Call + Put)
            c_g = b76.greeks(forward_curr, tranche.strike, tau_curr, dvol_curr, rate, "call")
            p_g = b76.greeks(forward_curr, tranche.strike, tau_curr, dvol_curr, rate, "put")

            # Net Delta for Short Straddle (Short Options -> Positive Hedge Required)
            tranche_delta = (c_g.delta + p_g.delta) * tranche.weight
            total_port_delta += tranche_delta

            # Delta Hedge P&L on Forward Movement
            hedge_pnl = tranche_delta * (forward_next - forward_curr)

            # Rehedge Friction (Fee paid on delta change)
            delta_change = abs(tranche_delta - tranche.prev_delta)
            rehedge_cost = delta_change * s_curr * perp_fee_rate

            daily_hedge_pnl += hedge_pnl
            daily_friction_cost += rehedge_cost
            tranche.prev_delta = tranche_delta

            # Close Tranche if 7-day maturity reached
            if tranche.days_held >= hold_days:
                tau_exit = (30.0 - hold_days) / 365.0
                forward_exit = next_spot * np.exp(rate * tau_exit)
                exec_exit_iv = dvol_curr + spread_haircut_vols

                exit_call = b76.price(forward_exit, tranche.strike, tau_exit, exec_exit_iv, rate, "call")
                exit_put = b76.price(forward_exit, tranche.strike, tau_exit, exec_exit_iv, rate, "put")
                repurchase_cost = (exit_call + exit_put) * tranche.weight

                tranche_option_pnl = tranche.collected_premium - repurchase_cost
                daily_option_pnl += tranche_option_pnl
                active_tranches.remove(tranche)

        daily_net_pnl = daily_option_pnl + daily_hedge_pnl - daily_friction_cost
        cum_pnl += daily_net_pnl
        daily_pnl_history.append(daily_net_pnl)
        portfolio_delta_history.append(total_port_delta)

    # Performance Metrics
    pnl_arr = np.array(daily_pnl_history)
    annualized_sharpe = (np.mean(pnl_arr) / np.std(pnl_arr)) * np.sqrt(365.0) if np.std(pnl_arr) > 0 else 0.0
    win_rate = np.mean(pnl_arr > 0)

    print(f"Evaluated Days:         {len(pnl_arr)}")
    print(f"Daily Win Rate:         {win_rate:.2%}")
    print(f"Cumulative Net P&L:     ${cum_pnl:,.2f} per 1 BTC straddle capacity")
    print(f"Mean Daily P&L:         ${np.mean(pnl_arr):,.2f}")
    print(f"Daily P&L StdDev:       ${np.std(pnl_arr):,.2f}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")

    # Plot Results
    plot_continuous_results(
        dates[burn_in : burn_in + len(pnl_arr)],
        prices[burn_in : burn_in + len(pnl_arr)],
        dvol_ivs[burn_in : burn_in + len(pnl_arr)],
        garch_forecasts,
        np.cumsum(pnl_arr),
    )


def plot_continuous_results(eval_dates, eval_spots, eval_dvol, eval_garch, cum_pnl):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")

    # Panel 1: Continuous Portfolio P&L vs Spot
    ax1.plot(eval_dates, cum_pnl, color="#1a1a1a", lw=1.8, label="Continuous Portfolio P&L ($)")
    ax1.set_ylabel("Net P&L (USD)")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_spot = ax1.twinx()
    ax1_spot.plot(eval_dates, eval_spots, color="#777777", lw=1.2, ls="--", label="BTC Spot ($)")
    ax1_spot.set_ylabel("BTC Spot (USD)")
    ax1.set_title("Continuous 7-Tranche VRP Harvesting Strategy (Delta-Hedged)")

    # Panel 2: DVOL vs GARCH
    ax2.plot(eval_dates, [d * 100 for d in eval_dvol], color="#1f77b4", lw=1.4, label="Deribit DVOL")
    ax2.plot(eval_dates, [g * 100 for g in eval_garch], color="#d62728", lw=1.4, ls="--", label="GJR-GARCH Forecast (30d)")
    ax2.set_ylabel("Volatility (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    tick_step = max(1, len(eval_dates) // 8)
    ax2.set_xticks(eval_dates[::tick_step])
    ax2.tick_params(axis="x", rotation=20)

    fig.savefig("vrp_harvest_continuous.png", dpi=180)
    plt.close(fig)
    print("\nSaved updated chart to: vrp_harvest_continuous.png")


if __name__ == "__main__":
    run_continuous_vrp_backtest()