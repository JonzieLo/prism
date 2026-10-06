from dataclasses import dataclass

@dataclass(frozen=True)
class SurfaceFilterPolicy:
    max_relative_spread: float | None = None
    minimum_points_per_expiry: int = 5