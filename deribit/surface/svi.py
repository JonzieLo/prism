from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Guards a division, not a modelling choice. Small enough that it never
# perturbs a real calibration, large enough to keep 1/sigma finite.
_VOLATILITY_FLOOR = 1e-8
_VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class SVIParameters:
    """Raw-SVI total-variance parameters."""

    a: float
    b: float
    rho: float
    m: float
    eta: float

    def validate(self) -> None:
        values = np.array(
            [self.a, self.b, self.rho, self.m, self.eta],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("SVI parameters must be finite")
        if self.b <= 0.0:
            raise ValueError("SVI b must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("SVI rho must lie strictly between -1 and 1")
        if self.eta <= 0.0:
            raise ValueError("SVI eta must be positive")
        if self.minimum_total_variance < 0.0:
            raise ValueError(
                "SVI minimum total variance must be non-negative"
            )

    @property
    def minimum_total_variance(self) -> float:
        return self.a + self.b * self.eta * np.sqrt(
            1.0 - self.rho * self.rho
        )

    @property
    def left_wing_slope(self) -> float:
        return self.b * (1.0 - self.rho)

    @property
    def right_wing_slope(self) -> float:
        return self.b * (1.0 + self.rho)

    def total_variance(
        self,
        k: ArrayLike,
    ) -> NDArray[np.float64]:
        values = np.asarray(k, dtype=float)
        centered = values - self.m
        return self.a + self.b * (
            self.rho * centered
            + np.sqrt(centered * centered + self.eta * self.eta)
        )

    def first_derivative(
        self,
        k: ArrayLike,
    ) -> NDArray[np.float64]:
        values = np.asarray(k, dtype=float)
        centered = values - self.m
        root = np.sqrt(centered * centered + self.eta * self.eta)
        return self.b * (self.rho + centered / root)

    def second_derivative(
        self,
        k: ArrayLike,
    ) -> NDArray[np.float64]:
        values = np.asarray(k, dtype=float)
        centered = values - self.m
        denominator = (
            centered * centered + self.eta * self.eta
        ) ** 1.5
        return self.b * self.eta * self.eta / denominator

    def implied_vol(
        self,
        k: ArrayLike,
        tau: float,
    ) -> NDArray[np.float64]:
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        variance = self.total_variance(k)
        if np.any(variance < 0.0):
            raise ValueError("SVI produced negative total variance")
        return np.sqrt(variance / tau)


    ###Dyanmics
    def dvol_dk(
        self,
        k: ArrayLike,
        tau: float,
    ) -> NDArray[np.float64]:
        # Floor the vol before dividing. implied_vol already rejects negative
        # variance, but a calibration sitting exactly on w = 0 would otherwise
        # produce inf here rather than a large finite slope.
        vol = np.maximum(self.implied_vol(k, tau), _VOLATILITY_FLOOR)
        return self.first_derivative(k) / (2.0 * tau * vol)

    def standardised_skew(
        self,
        k: ArrayLike,
        tau: float,
    ) -> NDArray[np.float64]:
        """sqrt(tau) * d(sigma)/dk.
 
        Raw d(sigma)/dk carries a 1/sqrt(tau) scaling, so short expiries
        always look steeper regardless of whether the market's skew view
        changed. Multiply it out before comparing across the term structure.
        """
        return np.sqrt(tau) * self.dvol_dk(k, tau)

    def dvol_dlnspot_fixed_strike(
        self,
        k: ArrayLike,
        tau: float,
        *,
        skew_stickiness_ratio: float = 0.0,
        forward_spot_elasticity: float = 1.0,
    ) -> NDArray[np.float64]:
        """d(sigma_K) / d(ln S) at fixed strike, under a stickiness rule.
 
        Let R be the skew-stickiness ratio, defined by
            d(sigma_ATM) / d(ln S) = R * (d sigma / dk)|_{k=0}
 
        Modelling the response as a translation of the smile in k,
            sigma_new(k) = sigma_old(k + R * delta),      delta = d ln S
 
        and noting that a fixed strike moves to k - eps*delta in the new
        forward's coordinates, the fixed-strike response is
 
            d(sigma_K) / d(ln S) = eps * (R - 1) * d(sigma)/dk
 
        R = 0   sticky delta / sticky moneyness. Smile frozen in k, so the
                whole smile translates with the forward and fixed-strike
                vols move the most.
        R = 1   sticky strike. Fixed-strike vols do not move at all.
        R ~ 1.5 typical equity-index regime; ATM vol over-reacts relative
                to sticky strike.
 
        eps = d ln F / d ln S. Equal to 1 under deterministic rates and
        no dividend/funding response, which for BTC perps-funded forwards
        is an approximation, not an identity.
        """
        if not np.isfinite(skew_stickiness_ratio):
            raise ValueError("skew_stickiness_ratio must be finite")
        if not np.isfinite(forward_spot_elasticity):
            raise ValueError("forward_spot_elasticity must be finite")
        return (
            forward_spot_elasticity
            * (skew_stickiness_ratio - 1.0)
            * self.dvol_dk(k, tau)
        )

    def dvol_dforward_fixed_strike(
        self,
        k: ArrayLike,
        forward: float,
        tau: float,
        skew_stickiness_ratio: float = 0.0,
    ) -> NDArray[np.float64]:
        if not np.isfinite(forward) or forward <= 0.0:
            raise ValueError("forward must be finite and positive")
        return (skew_stickiness_ratio - 1.0) * self.dvol_dk(k, tau) / forward

    def dvol_dspot_fixed_strike(
        self,
        k: ArrayLike,
        spot: float,
        tau: float,
        *,
        skew_stickiness_ratio: float = 0.0,
        forward_spot_elasticity: float = 1.0, ### d ln(F) / d ln(S)
    ) -> NDArray[np.float64]:
        if not np.isfinite(spot) or spot <= 0.0:
            raise ValueError("spot must be finite and positive")
        
        return (
            self.dvol_dlnspot_fixed_strike(
                k,
                tau,
                skew_stickiness_ratio=skew_stickiness_ratio,
                forward_spot_elasticity=forward_spot_elasticity,
            )
            / spot
        )

    ### Shape diagnostics
    #
    # These are the quantities to compare across expiries and snapshots.
    # The raw five parameters move together under recalibration, so reading
    # any one of them in isolation is unreliable.

    @property
    def minimum_total_variance_k(self) -> float:
        """k* where total variance is minimised.

            k* = m - rho * eta / sqrt(1 - rho^2)

        NOT m. The two coincide only at rho = 0, and the gap is material at
        the rho values a BTC smile actually fits: at rho = -0.3, eta = 0.2
        the minimum sits 0.063 to the right of m.
        """
        return float(
            self.m
            - self.rho * self.eta / np.sqrt(1.0 - self.rho * self.rho)
        )

    @property
    def atm_curvature(self) -> float:
        """w''(0), the total-variance curvature at the money.

        Peak curvature is b / eta and occurs at k = m, so b / eta is an
        upper bound on this, not an equivalent of it.
        """
        return float(self.second_derivative(0.0))

    def atm_total_variance(self) -> float:
        """w(0). Multiply by nothing; divide by tau for ATM variance."""
        return float(self.total_variance(0.0))

    def vega(
        self,
        k: ArrayLike,
        forward: float,
        tau: float,
        *,
        discount_factor: float = 1.0,
    ) -> NDArray[np.float64]:
        """Black-76 vega, d(price) / d(sigma), in price units per 1.0 of vol.

        Undiscounted by default. Pass discount_factor = exp(-r * tau) to
        match deribit.pricing.Black76Model, which discounts. For price change
        per vol POINT rather than per unit vol, divide the result by 100.

        Written in terms of total variance w rather than sigma and tau
        separately, since that is what SVI actually holds:

            d1   = -k / sqrt(w) + sqrt(w) / 2
            vega = DF * F * phi(d1) * sqrt(tau)
        """
        if not np.isfinite(forward) or forward <= 0.0:
            raise ValueError("forward must be finite and positive")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if not np.isfinite(discount_factor) or discount_factor <= 0.0:
            raise ValueError("discount_factor must be finite and positive")
        values = np.asarray(k, dtype=float)
        variance = np.maximum(self.total_variance(values), _VARIANCE_FLOOR)
        root_variance = np.sqrt(variance)
        d1 = -values / root_variance + 0.5 * root_variance
        pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
        return discount_factor * forward * pdf * np.sqrt(tau)
