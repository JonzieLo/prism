from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq

from .base import CallPut, Greeks, OptionModel, parse_cp
from .black_scholes import Black76Model


@dataclass(frozen=True)
class MertonJumpParams:
    intensity: float = 2.0      # lambda: expected jump events per year
    jump_mean: float = -0.05    # mu_J: mean of log-jump size (-ve in crypto for crash risk)
    jump_std: float = 0.15      # sigma_J: stdev of log-jump size

    def validate(self) -> None:
        if self.intensity < 0.0:
            raise ValueError(f"Jump intensity lambda must be non-negative, got {self.intensity}")
        if self.jump_std <= 0.0:
            raise ValueError(f"Jump std sigma_J must be positive, got {self.jump_std}")
        if not (math.isfinite(self.intensity) and math.isfinite(self.jump_mean) and math.isfinite(self.jump_std)):
            raise ValueError("All jump parameters must be finite")

    @property
    def expected_jump_size(self) -> float:
        """k_J = E[e^Y - 1]"""
        return math.exp(self.jump_mean + 0.5 * self.jump_std ** 2) - 1.0


class MertonModel(OptionModel):
    def __init__(
        self,
        jump_params: MertonJumpParams = MertonJumpParams(),
        max_jumps: int = 40,
        truncation_tol: float = 1e-14,
    ):
        jump_params.validate()
        self.jump_params = jump_params
        self.max_jumps = max_jumps
        self.truncation_tol = truncation_tol
        self._b76 = Black76Model()

    def price(
        self,
        forward: float,
        strike: float,
        tau: float,
        vol: float,
        rate: float,
        cp: CallPut,
    ) -> float:
        """
        Calculates fair price under Merton Jump Diffusion.
        :param vol: continuous diffusion volatility (sigma)
        """
        if tau <= 0.0 or vol <= 0.0:
            return self._b76.price(forward, strike, tau, vol, rate, cp)

        lam = self.jump_params.intensity
        if lam == 0.0:
            return self._b76.price(forward, strike, tau, vol, rate, cp)

        k_j = self.jump_params.expected_jump_size
        mu_j = self.jump_params.jump_mean
        sigma_j2 = self.jump_params.jump_std ** 2

        # Martingale
        compensator = math.exp(-lam * k_j * tau)

        total_price = 0.0
        poisson_prob = math.exp(-lam * tau)

        for n in range(self.max_jumps):
            sigma_n = math.sqrt(vol * vol + (n * sigma_j2) / tau)
            f_n = forward * compensator * math.exp(n * (mu_j + 0.5 * sigma_j2))

            b76_price = self._b76.price(f_n, strike, tau, sigma_n, rate, cp)
            term = poisson_prob * b76_price
            total_price += term

            if term < self.truncation_tol * max(total_price, 1e-6) and n > 5:
                break
            poisson_prob *= (lam * tau) / (n + 1)

        return total_price

    def greeks(
        self,
        forward: float,
        strike: float,
        tau: float,
        vol: float,
        rate: float,
        cp: CallPut,
    ) -> Greeks:
        if tau <= 0.0 or vol <= 0.0:
            return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, vanna=0.0, vomma=0.0)

        h_f = max(forward * 1e-5, 1e-4)
        h_v = max(vol * 1e-5, 1e-6)
        h_t = max(tau * 1e-5, 1e-6)
        h_r = 1e-5

        p = self.price(forward, strike, tau, vol, rate, cp)
        p_f_up = self.price(forward + h_f, strike, tau, vol, rate, cp)
        p_f_dn = self.price(forward - h_f, strike, tau, vol, rate, cp)
        p_v_up = self.price(forward, strike, tau, vol + h_v, rate, cp)
        p_v_dn = self.price(forward, strike, tau, vol - h_v, rate, cp)
        p_t_up = self.price(forward, strike, tau + h_t, rate, cp)
        p_t_dn = self.price(forward, strike, tau - h_t, rate, cp)
        p_r_up = self.price(forward, strike, tau, vol, rate + h_r, cp)
        p_r_dn = self.price(forward, strike, tau, vol, rate - h_r, cp)

        delta = (p_f_up - p_f_dn) / (2.0 * h_f)
        gamma = (p_f_up - 2.0 * p + p_f_dn) / (h_f * h_f)
        vega = (p_v_up - p_v_dn) / (2.0 * h_v)
        theta = -(p_t_up - p_t_dn) / (2.0 * h_t)
        rho = (p_r_up - p_r_dn) / (2.0 * h_r)

        # Cross Greeks
        p_fv_up = self.price(forward + h_f, strike, tau, vol + h_v, rate, cp)
        p_fv_dn = self.price(forward - h_f, strike, tau, vol + h_v, rate, cp)
        delta_v_up = (p_fv_up - p_fv_dn) / (2.0 * h_f)
        vanna = (delta_v_up - delta) / h_v
        vomma = (p_v_up - 2.0 * p + p_v_dn) / (h_v * h_v)

        return Greeks(
            delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, vanna=vanna, vomma=vomma
        )

    def implied_vol(
        self,
        price: float,
        forward: float,
        strike: float,
        tau: float,
        rate: float,
        cp: CallPut,
    ) -> float:
        df = math.exp(-rate * tau)
        is_call = parse_cp(cp) == 1
        intrinsic = df * max(0.0, forward - strike) if is_call else df * max(0.0, strike - forward)
        ceiling = df * (forward if is_call else strike)

        if price <= intrinsic:
            raise ValueError(f"Price {price} is below intrinsic {intrinsic}")
        if price >= ceiling:
            raise ValueError(f"Price {price} exceeds ceiling bound {ceiling}")

        def obj(v: float) -> float:
            return self.price(forward, strike, tau, v, rate, cp) - price

        try:
            return brentq(obj, 1e-4, 5.0, xtol=1e-10, maxiter=100)
        except Exception as exc:
            raise ValueError(f"Merton implied vol solver failed for price {price}") from exc

    def b76_implied_vol(
        self,
        forward: float,
        strike: float,
        tau: float,
        vol: float,
        rate: float,
        cp: CallPut,
    ) -> float:
        merton_px = self.price(forward, strike, tau, vol, rate, cp)
        return self._b76.implied_vol(merton_px, forward, strike, tau, rate, cp)