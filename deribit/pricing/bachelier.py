import math

from scipy.optimize import brentq
from scipy.stats import norm

from .base import CallPut, Greeks, OptionModel, parse_cp


MAX_NORMAL_VOL_MULTIPLE = 5.0     # initial upper bracket: sigma_N <= 5 * F
MIN_NORMAL_VOL_FRACTION = 1e-8    # lower bracket: sigma_N >= 1e-8 * F
BRACKET_GROWTH = 4.0              # expansion factor if the upper bracket is too low
BRACKET_MAX_DOUBLINGS = 20
INTRINSIC_ULP_TOL = 8.0           # "at intrinsic" == within 8 float64 steps
NEWTON_MAX_ITER = 20
NEWTON_REL_STEP_TOL = 1e-15       # meaningful at any sigma_N scale
MIN_VEGA = 1e-12


class NoTimeValueError(ValueError):
    """
    Raised when a quote sits at its intrinsic value to within float64 precision.

    Subclasses ValueError so forward models can count it as a filtered quote rather than treating it as bad data.
    """


class BachelierModel(OptionModel):
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
        Bachelier option price off the forward F.

        :param vol: normal volatility sigma_N, in price units per sqrt(year).
        """
        if tau <= 0.0 or vol <= 0.0:
            is_call = parse_cp(cp) == 1
            intrinsic = max(0.0, forward - strike) if is_call else max(0.0, strike - forward)
            return intrinsic * math.exp(-rate * max(0.0, tau))

        df = math.exp(-rate * tau)
        std_dev = vol * math.sqrt(tau)
        d = (forward - strike) / std_dev
        is_call = parse_cp(cp) == 1

        if is_call:
            return df * ((forward - strike) * norm.cdf(d) + std_dev * norm.pdf(d))
        return df * ((strike - forward) * norm.cdf(-d) + std_dev * norm.pdf(d))

    def greeks(
        self,
        forward: float,
        strike: float,
        tau: float,
        vol: float,
        rate: float,
        cp: CallPut,
    ) -> Greeks:
        """
        Analytic Bachelier Greeks off the forward F.

        vega, vanna and vomma are derivatives with respect to sigma_N, so to compare to Black-Scholes we have to convert sigma_N ~ sigma_LN * F.
        """
        if tau <= 0.0 or vol <= 0.0:
            return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, vanna=0.0, vomma=0.0)

        df = math.exp(-rate * tau)
        sqrt_tau = math.sqrt(tau)
        std_dev = vol * sqrt_tau
        d = (forward - strike) / std_dev

        n_d = norm.cdf(d)
        pdf_d = norm.pdf(d)
        is_call = parse_cp(cp) == 1

        delta = df * n_d if is_call else -df * norm.cdf(-d)
        gamma = df * pdf_d / std_dev
        vega = df * sqrt_tau * pdf_d

        theta_decay = -0.5 * df * vol * pdf_d / sqrt_tau
        if is_call:
            theta = theta_decay + rate * df * ((forward - strike) * n_d + std_dev * pdf_d)
            rho = -tau * self.price(forward, strike, tau, vol, rate, "call")
        else:
            theta = theta_decay + rate * df * ((strike - forward) * norm.cdf(-d) + std_dev * pdf_d)
            rho = -tau * self.price(forward, strike, tau, vol, rate, "put")

        vanna = -df * d * pdf_d / vol
        vomma = df * sqrt_tau * d * d * pdf_d / vol

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
        """
        Invert a Bachelier price for normal volatility sigma_N.

        :raises NoTimeValueError: price is at intrinsic to float64 precision.
        :raises ValueError: price is below intrinsic, or the solver failed.
        """
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        if forward <= 0.0:
            raise ValueError(f"forward must be positive, got {forward}")

        df = math.exp(-rate * tau)
        is_call = parse_cp(cp) == 1
        intrinsic = df * max(0.0, forward - strike) if is_call else df * max(0.0, strike - forward)

        # "At intrinsic" means within a few steps of float64 resolution at the scale of the intrinsic, not within some fixed dollar amount. 
        # A fixed 1e-10 would be 750,000 ulp at a 55,000 intrinsic and would throw away perfectly recoverable quotes.
        tol = max(INTRINSIC_ULP_TOL * math.ulp(abs(intrinsic)), 1e-300)

        if price < intrinsic - tol:
            raise ValueError(
                f"Price {price!r} is below intrinsic {intrinsic!r} "
                f"by {intrinsic - price:.6e} (tolerance {tol:.3e})"
            )
        if abs(price - intrinsic) <= tol:
            raise NoTimeValueError(
                f"Price {price!r} is at intrinsic {intrinsic!r} to float64 precision "
                f"(tolerance {tol:.3e}); sigma_N is not recoverable from this quote"
            )

        def obj(v: float) -> float:
            return self.price(forward, strike, tau, v, rate, cp) - price

        lo = MIN_NORMAL_VOL_FRACTION * forward
        hi = MAX_NORMAL_VOL_MULTIPLE * forward

        for _ in range(BRACKET_MAX_DOUBLINGS):
            if obj(hi) >= 0.0:
                break
            hi *= BRACKET_GROWTH
        else:
            raise ValueError(
                f"Price {price!r} exceeds the Bachelier price at sigma_N = {hi!r}; "
                f"quote is not attainable under this model"
            )

        v = self._seed(price, intrinsic, tau, df)
        v = min(max(v, lo), hi)

        for _ in range(NEWTON_MAX_ITER):
            vega = self.greeks(forward, strike, tau, v, rate, cp).vega
            if vega < MIN_VEGA:
                break

            step = (self.price(forward, strike, tau, v, rate, cp) - price) / vega
            if abs(step) < NEWTON_REL_STEP_TOL * max(abs(v), 1.0):
                return v - step

            v -= step
            if not (lo < v < hi):
                break

        # Brent's Method on the bracketed root
        try:
            return brentq(obj, lo, hi, rtol=8.9e-16, maxiter=200)
        except Exception as exc:
            raise ValueError(
                f"Failed to converge Bachelier implied vol for price {price!r} "
                f"(F={forward!r}, K={strike!r}, tau={tau!r}, cp={cp!r})"
            ) from exc

    @staticmethod
    def _seed(price: float, intrinsic: float, tau: float, df: float) -> float:
        """
        Scale-free starting guess from the at-the-money Bachelier identity

            P_atm = df * sigma_N * sqrt(tau / (2*pi))   =>   sigma_N = P * sqrt(2*pi/tau) / df

        which is EXACT when K == F and a usable order-of-magnitude guess elsewhere.
        Driven by time value, not total price, so a deep in-the-money quote does not produce a seed inflated by its intrinsic.
        Carries no currency constant, so it works unchanged on BTC, ETH, SOL or an index.
        """
        time_value = max(price - intrinsic, 0.0)
        base = time_value if time_value > 0.0 else price
        return max(base, 1e-300) * math.sqrt(2.0 * math.pi / tau) / df
