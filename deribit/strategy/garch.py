from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class GARCHParams:
    omega: float
    alpha: float
    beta: float
    gamma: float = 0.0
    mu: float = 0.0

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta + 0.5 * self.gamma

    @property
    def unconditional_variance(self) -> float:
        if self.persistence >= 1.0:
            return float("inf")
        return self.omega / (1.0 - self.persistence)

    @property
    def unconditional_vol_annualized(self) -> float:
        var = self.unconditional_variance
        return math.sqrt(var * 365.0) if math.isfinite(var) else float("nan")


@dataclass(frozen=True)
class VRPAnalysis:
    horizon_days: float
    implied_vol: float
    garch_forecast_vol: float
    vrp: float  # IV - GARCH_RV
    rich_cheap_status: str  # 'OVERPRICED_IV' or 'UNDERPRICED_IV'


class GJR_GARCH:

    def __init__(self, trading_days_per_year: float = 365.0):
        self.annualization = trading_days_per_year

    def fit(self, returns: Sequence[float] | np.ndarray, fit_asymmetry: bool = True) -> GARCHParams:
        """
        Fits GJR-GARCH(1,1) via Maximum Likelihood Estimation (MLE)
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < 30:
            raise ValueError(f"Need at least 30 observations to fit GARCH, got {len(r)}")

        sample_var = float(np.var(r))
        mu_init = float(np.mean(r))

        init_params = [sample_var * 0.05, 0.08, 0.85, 0.04 if fit_asymmetry else 0.0, mu_init]

        gamma_bound = (0.0, 0.5) if fit_asymmetry else (0.0, 0.0)
        bounds = [(1e-7, sample_var), (1e-4, 0.4), (0.4, 0.98), gamma_bound, (-0.05, 0.05)]

        def neg_log_likelihood(params: list[float]) -> float:
            omega, alpha, beta, gamma, mu = params
            if alpha + beta + 0.5 * gamma >= 0.999:
                return 1e10

            n = len(r)
            sig2 = np.zeros(n)
            sig2[0] = sample_var
            eps = r - mu

            for t in range(1, n):
                leverage = gamma * (eps[t - 1] ** 2) if eps[t - 1] < 0.0 else 0.0
                sig2[t] = omega + alpha * (eps[t - 1] ** 2) + leverage + beta * sig2[t - 1]

            sig2 = np.maximum(sig2, 1e-9)
            ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sig2) + (eps ** 2) / sig2)
            return float(-ll)

        res = minimize(
            neg_log_likelihood,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )

        p = res.x
        return GARCHParams(omega=p[0], alpha=p[1], beta=p[2], gamma=p[3], mu=p[4])

    def filter_variance(self, returns: Sequence[float] | np.ndarray, params: GARCHParams) -> np.ndarray:
        r = np.asarray(returns, dtype=float)
        n = len(r)
        sig2 = np.zeros(n)
        sig2[0] = params.unconditional_variance if math.isfinite(params.unconditional_variance) else np.var(r)
        eps = r - params.mu

        for t in range(1, n):
            leverage = params.gamma * (eps[t - 1] ** 2) if eps[t - 1] < 0.0 else 0.0
            sig2[t] = params.omega + params.alpha * (eps[t - 1] ** 2) + leverage + params.beta * sig2[t - 1]

        return sig2

    def forecast_realized_vol(
        self,
        params: GARCHParams,
        last_variance: float,
        last_shock: float,
        horizon_days: float,
    ) -> float:
        if horizon_days <= 0.0:
            raise ValueError(f"horizon_days must be positive, got {horizon_days}")

        persist = params.persistence
        v_bar = params.unconditional_variance
        h_int = max(1, int(round(horizon_days)))

        leverage = params.gamma * (last_shock ** 2) if last_shock < 0.0 else 0.0
        sig2_1 = params.omega + params.alpha * (last_shock ** 2) + leverage + params.beta * last_variance

        cum_variance = 0.0
        for k in range(h_int):
            if persist < 1.0:
                expected_k = v_bar + (persist ** k) * (sig2_1 - v_bar)
            else:
                expected_k = sig2_1 + k * params.omega
            cum_variance += expected_k

        mean_daily_var = cum_variance / h_int
        return math.sqrt(mean_daily_var * self.annualization)

    def analyze_vrp(
        self,
        implied_vol: float,
        params: GARCHParams,
        last_variance: float,
        last_shock: float,
        horizon_days: float,
        threshold_bps: float = 200.0,
    ) -> VRPAnalysis:
        """
        Volatility Risk Premium (VRP = IV - GARCH_RV).
        Positive VRP = market IV is rich relative to forecasted realized volatility.
        """
        forecast_rv = self.forecast_realized_vol(params, last_variance, last_shock, horizon_days)
        vrp = implied_vol - forecast_rv

        thresh = threshold_bps / 10_000.0
        if vrp > thresh:
            status = "OVERPRICED_IV"
        elif vrp < -thresh:
            status = "UNDERPRICED_IV"
        else:
            status = "FAIR_VALUE"

        return VRPAnalysis(
            horizon_days=horizon_days,
            implied_vol=implied_vol,
            garch_forecast_vol=forecast_rv,
            vrp=vrp,
            rich_cheap_status=status,
        )