import math
import sys
import numpy as np
from scipy.optimize import brentq
from .bachelier import NoTimeValueError
from .base import CallPut, Greeks, OptionModel, parse_cp
from .black_scholes import Black76Model

FD_REL_VOL = 1e-4
FD_REL_SPOT = 1e-4
FD_REL_RATE = 1e-4

INTRINSIC_ULP_TOL = 8.0

class BinomialModel(OptionModel):
    """
    Cox-Ross-Rubinstein binomial model for European options.
    """

    def __init__(self, steps: int = 200):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"Steps must be a positive integer, got {steps!r}")
        self.steps = steps

    @staticmethod
    def _validate(forward: float, strike: float, tau: float, vol: float, rate: float) -> None:
        for name, value in (
            ("forward", forward),
            ("strike", strike),
            ("tau", tau),
            ("vol", vol),
            ("rate", rate),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if forward <= 0.0:
            raise ValueError(f"forward must be positive, got {forward!r}")
        if strike <= 0.0:
            raise ValueError(f"strike must be positive, got {strike!r}")
        if tau < 0.0:
            raise ValueError(f"tau must be non-negative, got {tau!r}")
        if vol < 0.0:
            raise ValueError(f"vol must be non-negative, got {vol!r}")
        
    def _tree(
        self,
        forward: float,
        strike: float,
        tau: float,
        vol: float,
        rate: float,
        sign: int,
        return_levels: bool = False,
    ) -> float:
        dt = tau / self.steps
        x = vol * math.sqrt(dt)
        max_log_node = math.log(forward) + self.steps * x
        if max_log_node > math.log(sys.float_info.max):
            raise ValueError("Tree contains a forward above float64 range")

        u = math.exp(x)
        p = 1.0 / (1.0 + u)
        if not 0.0 < p < 1.0:
            raise ValueError(f"Risk-neutral probability is not in (0, 1): {p!r}")

        step_df = math.exp(-rate * dt)
        if not math.isfinite(math.exp(-rate * tau)):
            raise ValueError("Discount factor is outside float64 range")

        j = np.arange(self.steps + 1, dtype=float)
        log_nodes = math.log(forward) + (self.steps - 2.0 * j) * x
        with np.errstate(under="ignore"):
            nodes = np.exp(log_nodes)
        values = np.maximum(sign * (nodes - strike), 0.0)
        level_2 = values.copy() if self.steps == 2 else None
        level_1 = values.copy() if self.steps == 1 else None

        for _ in range(self.steps):
            values = step_df * (p * values[:-1] + (1.0 - p) * values[1:])
            if len(values) == 3:
                level_2 = values.copy()
            elif len(values) == 2:
                level_1 = values.copy()

        root = float(values[0])
        if return_levels:
            return root, level_1, level_2
        return root
    
    def _delta(
        self,
        forward: float,
        strike: float,
        tau: float,
        vol: float,
        rate: float,
        sign: int,
    ) -> float:
        _, level_1, _ = self._tree(
            forward, strike, tau, vol, rate, sign, return_levels=True
        )
        dt = tau / self.steps
        u = math.exp(vol * math.sqrt(dt))
        d = 1.0 / u
        return float((level_1[0] - level_1[1]) / (forward * (u - d)))

    def price( self,
            forward: float,
            strike: float,
            tau: float,
            vol: float,
            rate: float,
            cp: CallPut
            ) -> float:
        self._validate(forward, strike, tau, vol, rate)
        sign = parse_cp(cp)
        intrinsic = max(sign * (forward - strike), 0.0)
        if tau == 0.0:
            return intrinsic
        if vol == 0.0:
            return math.exp(-rate * tau) * intrinsic
        return self._tree(forward, strike, tau, vol, rate, sign)

    def greeks(self,
            forward: float,
            strike: float,
            tau: float,
            vol: float,
            rate: float,
            cp: CallPut,
        ) -> Greeks:
        self._validate(forward, strike, tau, vol, rate)
        if self.steps < 2:
            raise ValueError(f"At least 2 steps needed, got {self.steps}.")
        if tau == 0.0 or vol == 0.0:
            raise ValueError("Greeks are not defined at Zero tau or Zero Volatility.")
        
        sign = parse_cp(cp)
        root, level_1, level_2 = self._tree(
            forward, strike, tau, vol, rate, sign, return_levels=True
        )
        dt = tau / self.steps
        u = math.exp(vol * math.sqrt(dt))
        d = 1.0 / u

        delta = float((level_1[0] - level_1[1]) / (forward * (u - d)))
        delta_up = (level_2[0] - level_2[1]) / (forward * u * (u - d))
        delta_down = (level_2[1] - level_2[2]) / (forward * d * (u - d))
        gamma = float(
            (delta_up - delta_down) / (0.5 * forward * (u * u - d * d))
        )
        theta = float((level_2[1] - root) / (2.0 * dt))
        rho = -tau * root

        eps = sys.float_info.epsilon
        h_first = vol * eps ** (1.0 / 3.0)
        h_second = vol * eps ** 0.25

        price_up = self._tree(forward, strike, tau, vol + h_first, rate, sign)
        price_down = self._tree(forward, strike, tau, vol - h_first, rate, sign)
        vega = (price_up - price_down) / (2.0 * h_first)

        delta_up_vol = self._delta(
            forward, strike, tau, vol + h_first, rate, sign
        )
        delta_down_vol = self._delta(
            forward, strike, tau, vol - h_first, rate, sign
        )
        vanna = (delta_up_vol - delta_down_vol) / (2.0 * h_first)

        price_up_2 = self._tree(
            forward, strike, tau, vol + h_second, rate, sign
        )
        price_down_2 = self._tree(
            forward, strike, tau, vol - h_second, rate, sign
        )
        vomma = (price_up_2 - 2.0 * root + price_down_2) / (
            h_second * h_second
        )

        return Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            vanna=vanna,
            vomma=vomma,
        )


    def implied_vol(self,
            price: float,
            forward: float,
            strike: float,
            tau: float,
            rate: float,
            cp: CallPut,
        ) -> float:
        self._validate(forward, strike, tau, 0.0, rate)
        if tau == 0.0:
            raise ValueError(f"Tau must be positive, got {tau!r}")
        if not math.isfinite(price) or price < 0.0:
            raise ValueError(f"Price must be finite and non-negative, got {price!r}")

        sign = parse_cp(cp)
        df = math.exp(-rate * tau)
        intrinsic = df * max(sign * (forward - strike), 0.0)
        ceiling = df * (forward if sign == 1 else strike)

        if price < intrinsic:
            raise ValueError(f"Price {price!r} is below intrinsic {intrinsic!r}")
        if price == intrinsic:
            raise NoTimeValueError(
                f"Price {price!r} is at intrinsic; volatility is not recoverable"
            )
        if price >= ceiling:
            raise ValueError(f"Price {price!r} is at or above ceiling {ceiling!r}")

        def objective(candidate: float) -> float:
            return self.price(forward, strike, tau, candidate, rate, cp) - price

        time_value = price - intrinsic
        hi = max(
            time_value * math.sqrt(2.0 * math.pi / tau) / (df * forward),
            math.sqrt(sys.float_info.epsilon / tau),
        )
        max_vol = (
            math.log(sys.float_info.max) - math.log(forward)
        ) / math.sqrt(tau * self.steps)

        while hi < max_vol and objective(hi) < 0.0:
            hi = min(2.0 * hi, max_vol)

        f_hi = objective(hi)
        if f_hi < 0.0:
            raise ValueError(
                f"Price {price!r} is not bracketed within the float64 CRR domain"
            )
        return brentq(objective, 0.0, hi)