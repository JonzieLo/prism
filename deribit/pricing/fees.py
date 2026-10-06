"""
Deribit Fees (Standard)
- Options: 0.03% of underlying index, capped at 12.5% of option premium
- Perpetuals/Futures: 0.035% of notional (taker), 0.015% (maker)
- Spot: 0.05% of notional (taker), 0.02% (maker)

Combo Discounts:
- Options: (CALL & PUT) Cheaper direction is 0
"""
from typing import Sequence

def calculate_option_fee(
    premium_usd: float,
    index_price: float,
    contracts: float = 1.0,
    is_taker: bool = True,
    cap_fraction: float = 0.125,
) -> float:
    if premium_usd <= 0.0 or index_price <= 0.0 or contracts <= 0.0:
        return 0.0

    rate = 0.0003
    uncapped_fee = rate * index_price
    capped_fee = min(uncapped_fee, cap_fraction * premium_usd)
    return float(capped_fee * contracts)


def calculate_option_combo_fee(
    buy_legs: Sequence[tuple[float, float]],   # (premium_usd, contracts)
    sell_legs: Sequence[tuple[float, float]],  # (premium_usd, contracts)
    index_price: float,
    is_taker: bool = True,
) -> float:
    buy_fees = sum(
        calculate_option_fee(prem, index_price, contracts=qty, is_taker=is_taker) for prem, qty in buy_legs
    )
    sell_fees = sum(
        calculate_option_fee(prem, index_price, contracts=qty, is_taker=is_taker) for prem, qty in sell_legs
    )
    if buy_fees > 0.0 and sell_fees > 0.0:
        return max(buy_fees, sell_fees)
    return buy_fees + sell_fees


def calculate_future_fee(
    future_price_usd: float,
    contracts: float = 1.0,
    is_taker: bool = True
) -> float:
    if future_price_usd <= 0.0 or contracts <= 0.0:
        return 0.0

    fee_rate = 0.00035 if is_taker else 0.00015
    return float(future_price_usd * contracts * fee_rate)