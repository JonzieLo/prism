import numpy as np
from deribit.strategy.garch import GJR_GARCH

# 1. Simulated daily log returns (or load real closes from SQLite/snapshots.db)
np.random.seed(42)
daily_returns = np.random.normal(loc=0.0005, scale=0.025, size=180)  # ~6 months

# 2. Fit GJR-GARCH(1,1)
engine = GJR_GARCH()
params = engine.fit(daily_returns)
print(f"GARCH Fitted: omega={params.omega:.2e}, alpha={params.alpha:.3f}, beta={params.beta:.3f}, persistence={params.persistence:.3f}")
print(f"Long-Run Unconditional Vol: {params.unconditional_vol_annualized * 100:.2f}%\n")

# 3. Filter variance path to get the latest state (t = now)
sig2_path = engine.filter_variance(daily_returns, params)
last_var = sig2_path[-1]
last_shock = daily_returns[-1] - params.mu

# 4. Compare a 30-day Deribit option (Implied Vol = 55%) against GARCH forecast
market_atm_iv = 0.55
analysis = engine.analyze_vrp(
    implied_vol=market_atm_iv,
    params=params,
    last_variance=last_var,
    last_shock=last_shock,
    horizon_days=30.0,
    threshold_bps=200.0,  # 2.0 vol points edge required
)

print("=== VOLATILITY RISK PREMIUM (VRP) ANALYSIS ===")
print(f"Horizon:            {analysis.horizon_days:.0f} days")
print(f"Market Implied Vol: {analysis.implied_vol * 100:.2f}%")
print(f"GARCH Forecast RV:  {analysis.garch_forecast_vol * 100:.2f}%")
print(f"VRP (IV - RV):      {analysis.vrp * 100:+.2f} vol points")
print(f"Trading Signal:     {analysis.rich_cheap_status}")