from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SurfaceExpiryContext:
    underlying_index: str
    expiration_timestamp: int
    index_price: float
    forward: float
    tau: float
    rate: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.index_price) or self.index_price <= 0.0:
            raise ValueError("index_price must be finite and positive")
        if not math.isfinite(self.forward) or self.forward <= 0.0:
            raise ValueError("forward must be finite and positive")
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive")