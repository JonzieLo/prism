from dataclasses import dataclass

from .base import Greeks


@dataclass(frozen=True)
class InverseGreeks:
    coin_price: float
    traditional_spot_delta: float
    coin_delta: float
    net_transaction_delta: float
    coin_gamma: float
    coin_vega: float
    coin_theta: float
    coin_rho: float
    coin_vanna: float
    coin_vomma: float


def from_forward_greeks(
    cash_price: float,
    forward_greeks: Greeks,
    index_price: float,
    forward: float,
) -> InverseGreeks:
    """Convert cash-valued forward Greeks to inverse coin units."""
    if index_price <= 0.0:
        raise ValueError(f"Index price must be positive, got {index_price}")
    if forward <= 0.0:
        raise ValueError(f"Forward must be positive, got {forward}")

    forward_per_index = forward / index_price
    spot_delta = forward_per_index * forward_greeks.delta
    spot_gamma = forward_per_index**2 * forward_greeks.gamma
    spot_vanna = forward_per_index * forward_greeks.vanna
    coin_price = cash_price / index_price
    net_transaction_delta = spot_delta - coin_price

    return InverseGreeks(
        coin_price=coin_price,
        traditional_spot_delta=spot_delta,
        coin_delta=net_transaction_delta / index_price,
        net_transaction_delta=net_transaction_delta,
        coin_gamma=(
            spot_gamma / index_price
            - 2.0 * spot_delta / index_price**2
            + 2.0 * cash_price / index_price**3
        ),
        coin_vega=forward_greeks.vega / index_price,
        coin_theta=forward_greeks.theta / index_price,
        coin_rho=forward_greeks.rho / index_price,
        coin_vanna=(
            spot_vanna / index_price
            - forward_greeks.vega / index_price**2
        ),
        coin_vomma=forward_greeks.vomma / index_price,
    )