from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
