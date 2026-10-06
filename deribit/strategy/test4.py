"""
PRISM Quantitative Options Strategy: Iteration 4 (V4).

Upgrades implemented:
1. Anti-Trough Realized Volatility: RV_effective = max(Parkinson_7d, Parkinson_30d).
2. Absolute Volatility Floor: Entry requires DVOL >= 0.35 (35%).
3. Macro Volatility Trend Filter: DVOL < SMA_20(DVOL).
4. Tranche-Level Stop-Loss: Hard cutoff if tranche loss exceeds 1.5x collected premium.
5. Continuous 1/7th daily rolling tranches with 3 bps taker fee rehedging friction.
"""

from datetime import datetime, timezone
import time
import httpx
import matplotlib.pyplot as plt
import numpy as np

from deribit.pricing.black_scholes import Black76Model


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

    # 2. Fetch Spot OHLC
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
    spot_highs = spot_res["high"]
    spot_lows = spot_res["low"]

    records = []
    for t, price, h, l in zip(spot_ticks, spot_closes, spot_highs, spot_lows):
        d_str = to_utc_date(t)
        if d_str in dvol_by_date:
            records.append((d_str, float(price), float(h), float(l), dvol_by_date[d_str]))

    print(f"-> Successfully aligned {len(records)} daily observations.\n")
    return records


def calculate_parkinson_rv(highs: np.ndarray, lows: np.ndarray, window: int = 30) -> float:
    """Calculates annualized Parkinson (High-Low) Realized Volatility over a rolling window."""
    if len(highs) < window:
        return 0.0
    h = highs[-window:]
    l = lows[-window:]
    l = np.where(l <= 0, h, l)
    log_hl_sq = (np.log(h / l)) ** 2
    sum_sq = np.sum(log_hl_sq)
    daily_var = sum_sq / (4.0 * np.log(2.0) * window)
    return float(np.sqrt(daily_var * 365.0))


class OptionTranche:
    """Represents 1/7th capacity short straddle tranche with MTM stop-loss tracking."""

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
        stop_loss_multiplier: float = 1.5,
    ):
        self.entry_idx = entry_idx
        self.entry_date = entry_date
        self.spot_0 = spot_0
        self.strike = spot_0  # ATM straddle
        self.hold_days = hold_days
        self.weight = capacity_weight
        self.rate = rate
        self.spread_haircut = spread_haircut_vols
        self.stop_loss_mult = stop_loss_multiplier

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
        self.accumulated_pnl = 0.0


def run_v4_vrp_backtest(
    days: int = 400,
    hold_days: int = 7,
    min_vrp_ratio: float = 1.15,           # ATM IV / max(RV_7, RV_30) >= 1.15
    min_dvol_floor: float = 0.35,          # Absolute DVOL Floor (35%)
    spread_haircut_vols: float = 0.02,      # 2 vol-pt spread execution haircut
    atm_skew_discount_vols: float = 0.03,  # DVOL - 3 vol pts for ATM IV
    perp_fee_bps: float = 3.0,             # 3 bps taker fee on futures rehedges
    stop_loss_mult: float = 1.5,           # Stop loss if tranche loss > 1.5x premium
):
    records = fetch_deribit_history(days=days)
    if len(records) < 180:
        print("Need at least 180 days of history.")
        return

    dates = [r[0] for r in records]
    prices = np.array([r[1] for r in records])
    highs = np.array([r[2] for r in records])
    lows = np.array([r[3] for r in records])
    dvol_ivs = np.array([r[4] for r in records])

    b76 = Black76Model()
    burn_in = 150
    rate = 0.03
    perp_fee_rate = perp_fee_bps / 10000.0

    print("=" * 80)
    print("=== PRISM V4: ANTI-TROUGH RV + VOL FLOOR + 20D SMA + STOP-LOSS ===")
    print(f"Capacity Split:      7 Daily Overlapping Tranches (1/7th allocation each)")
    print(f"VRP Entry Condition: ATM IV / max(RV_7d, RV_30d) >= {min_vrp_ratio:.2f}")
    print(f"Absolute Vol Floor:  DVOL >= {min_dvol_floor * 100:.1f}%")
    print(f"Vol Trend Filter:    DVOL < 20-Day SMA(DVOL)")
    print(f"Tranche Stop-Loss:   Close if Tranche Drawdown > {stop_loss_mult:.1f}x Collected Premium")
    print("=" * 80)

    active_tranches = []
    daily_pnl_history = []
    effective_rv_history = []
    stopped_out_count = 0

    cum_pnl = 0.0

    for i in range(burn_in, len(records) - 1):
        s_curr = prices[i]
        dvol_curr = dvol_ivs[i]

        # 1. Compute Parkinson RV over 7-day and 30-day windows
        rv_7d = calculate_parkinson_rv(highs[: i + 1], lows[: i + 1], window=7)
        rv_30d = calculate_parkinson_rv(highs[: i + 1], lows[: i + 1], window=30)
        
        # Anti-Trough Realized Volatility
        rv_effective = max(rv_7d, rv_30d)
        effective_rv_history.append(rv_effective)

        # 2. Compute 20-Day DVOL SMA
        dvol_sma20 = np.mean(dvol_ivs[max(0, i - 19) : i + 1])

        # 3. Estimate ATM IV
        atm_iv_curr = max(0.10, dvol_curr - atm_skew_discount_vols)

        # 4. Evaluate Multi-Condition Signal Filters
        vrp_ratio = atm_iv_curr / rv_effective if rv_effective > 0 else 0.0
        
        cond_vrp = vrp_ratio >= min_vrp_ratio
        cond_floor = dvol_curr >= min_dvol_floor
        cond_trend = dvol_curr < dvol_sma20

        entry_signal = cond_vrp and cond_floor and cond_trend

        if entry_signal:
            new_tranche = OptionTranche(
                entry_idx=i,
                entry_date=dates[i],
                spot_0=s_curr,
                atm_iv_0=atm_iv_curr,
                hold_days=hold_days,
                capacity_weight=1.0 / hold_days,
                rate=rate,
                spread_haircut_vols=spread_haircut_vols,
                stop_loss_multiplier=stop_loss_mult,
            )
            active_tranches.append(new_tranche)

        # 5. Process Active Tranches & Execute Daily Delta Rehedging
        daily_option_pnl = 0.0
        daily_hedge_pnl = 0.0
        daily_friction_cost = 0.0

        next_spot = prices[i + 1]

        for tranche in list(active_tranches):
            tranche.days_held += 1
            tau_curr = (30.0 - tranche.days_held) / 365.0
            forward_curr = s_curr * np.exp(rate * tau_curr)
            forward_next = next_spot * np.exp(rate * (tau_curr - 1.0 / 365.0))

            # Calculate Greeks for Short Straddle
            c_g = b76.greeks(forward_curr, tranche.strike, tau_curr, dvol_curr, rate, "call")
            p_g = b76.greeks(forward_curr, tranche.strike, tau_curr, dvol_curr, rate, "put")

            # Net Delta for Short Straddle
            tranche_delta = (c_g.delta + p_g.delta) * tranche.weight

            # Delta Hedge P&L on Forward Movement
            hedge_pnl = tranche_delta * (forward_next - forward_curr)

            # Rehedge Friction
            delta_change = abs(tranche_delta - tranche.prev_delta)
            rehedge_cost = delta_change * s_curr * perp_fee_rate

            daily_hedge_pnl += hedge_pnl
            daily_friction_cost += rehedge_cost
            tranche.prev_delta = tranche_delta

            # MTM Option Value
            exec_exit_iv = dvol_curr + spread_haircut_vols
            exit_call = b76.price(forward_next, tranche.strike, tau_curr, exec_exit_iv, rate, "call")
            exit_put = b76.price(forward_next, tranche.strike, tau_curr, exec_exit_iv, rate, "put")
            mtm_repurchase_cost = (exit_call + exit_put) * tranche.weight

            tranche_mtm_pnl = (tranche.collected_premium - mtm_repurchase_cost) + hedge_pnl - rehedge_cost
            tranche.accumulated_pnl += tranche_mtm_pnl

            # CHECK STOP-LOSS CONDITION OR EXPIRY
            stop_condition = tranche.accumulated_pnl < -(stop_loss_mult * tranche.collected_premium)
            expiry_condition = tranche.days_held >= hold_days

            if stop_condition or expiry_condition:
                if stop_condition and not expiry_condition:
                    stopped_out_count += 1

                tranche_option_pnl = tranche.collected_premium - mtm_repurchase_cost
                daily_option_pnl += tranche_option_pnl
                active_tranches.remove(tranche)

        daily_net_pnl = daily_option_pnl + daily_hedge_pnl - daily_friction_cost
        cum_pnl += daily_net_pnl
        daily_pnl_history.append(daily_net_pnl)

    # Performance Metrics
    pnl_arr = np.array(daily_pnl_history)
    annualized_sharpe = (np.mean(pnl_arr) / np.std(pnl_arr)) * np.sqrt(365.0) if np.std(pnl_arr) > 0 else 0.0
    win_rate = np.mean(pnl_arr > 0)

    print(f"Evaluated Days:         {len(pnl_arr)}")
    print(f"Daily Win Rate:         {win_rate:.2%}")
    print(f"Tranches Stopped Out:   {stopped_out_count}")
    print(f"Cumulative Net P&L:     ${cum_pnl:,.2f} per 1 BTC straddle capacity")
    print(f"Mean Daily P&L:         ${np.mean(pnl_arr):,.2f}")
    print(f"Daily P&L StdDev:       ${np.std(pnl_arr):,.2f}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")

    # Plot Results
    plot_v4_results(
        dates[burn_in : burn_in + len(pnl_arr)],
        prices[burn_in : burn_in + len(pnl_arr)],
        dvol_ivs[burn_in : burn_in + len(pnl_arr)],
        effective_rv_history,
        np.cumsum(pnl_arr),
    )


def plot_v4_results(eval_dates, eval_spots, eval_dvol, eval_rv_eff, cum_pnl):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")

    # Panel 1: Continuous Portfolio P&L vs Spot
    ax1.plot(eval_dates, cum_pnl, color="#1a1a1a", lw=1.8, label="PRISM V4 Continuous P&L ($)")
    ax1.set_ylabel("Net P&L (USD)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_spot = ax1.twinx()
    ax1_spot.plot(eval_dates, eval_spots, color="#777777", lw=1.2, ls="--", label="BTC Spot ($)")
    ax1_spot.set_ylabel("BTC Spot (USD)", fontweight="bold")
    ax1.set_title("PRISM V4: Anti-Trough RV + Vol Floor + 20d SMA + Stop-Loss", fontweight="bold")

    # Panel 2: DVOL vs Effective RV
    ax2.plot(eval_dates, [d * 100 for d in eval_dvol], color="#1f77b4", lw=1.4, label="Deribit DVOL (30d IV)")
    ax2.plot(eval_dates, [r * 100 for r in eval_rv_eff], color="#2ca02c", lw=1.4, ls="-", label="Effective RV (max 7d/30d)")
    ax2.axhline(35, color="gray", linestyle=":", label="35% DVOL Floor")
    ax2.set_ylabel("Volatility (%)", fontweight="bold")
    ax2.set_xlabel("Date", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    tick_step = max(1, len(eval_dates) // 8)
    ax2.set_xticks(eval_dates[::tick_step])
    ax2.tick_params(axis="x", rotation=20)

    fig.savefig("vrp_harvest_v4.png", dpi=180)
    plt.close(fig)
    print("\nSaved updated chart to: vrp_harvest_v4.png")


if __name__ == "__main__":
    run_v4_vrp_backtest()