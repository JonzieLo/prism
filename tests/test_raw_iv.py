import math
import pytest
from deribit.chain import OptionQuote
from deribit.forwards import ExpiryForward
from deribit.pricing import Black76Model
from deribit.surface.raw_iv import build_raw_iv_points


def _quotes() -> tuple[list[OptionQuote], ExpiryForward]:
    index = 65_000.0
    forward = 66_000.0
    tau = 0.4
    rate = math.log(forward / index) / tau
    expiry = 1_800_000_000_000
    model = Black76Model()
    quotes = []
    source_id = 1
    for strike in (55_000.0, 65_000.0, 75_000.0):
        for option_type in ("call", "put"):
            mid_usd = model.price(
                forward, strike, tau, 0.55, rate, option_type
            )
            mid_coin = mid_usd / index
            quotes.append(
                OptionQuote(
                    source_row_id=source_id,
                    instrument_name=(
                        f"BTC-TEST-{strike:.0f}-{option_type[0].upper()}"
                    ),
                    option_type=option_type,
                    strike=strike,
                    expiration_timestamp=expiry,
                    underlying_index="BTC-TEST",
                    settlement_currency="BTC",
                    contract_size=1.0,
                    index_price=index,
                    tau=tau,
                    bid_coin=0.99 * mid_coin,
                    ask_coin=1.01 * mid_coin,
                    bid_amount=1.0,
                    ask_amount=1.0,
                    mark_coin=mid_coin,
                    deribit_mark_iv=None,
                    open_interest=10.0,
                    volume=1.0,
                    last_coin=mid_coin,
                )
            )
            source_id += 1

    expiry_forward = ExpiryForward(
        expiration_timestamp=expiry,
        underlying_index="BTC-TEST",
        implied_forward=forward,
        dispersion_mad=0.0,
        dispersion_iqr=0.0,
        pair_count=3,
        best_synthetic_buy=None,
        best_synthetic_buy_strike=None,
        best_synthetic_sell=None,
        best_synthetic_sell_strike=None,
    )
    return quotes, expiry_forward


@pytest.mark.parametrize(
    "model_name",
    ["black_scholes", "black76", "binomial", "inverse"],
)
def test_raw_lognormal_iv_prototype_recovers_one_point_per_strike(
    model_name,
):
    quotes, expiry_forward = _quotes()
    result = build_raw_iv_points(
        7,
        quotes,
        [expiry_forward],
        model_name,
        binomial_steps=100,
    )

    assert result.snapshot_id == 7
    assert len(result.points) == 3
    assert not result.drops
    assert {point.strike for point in result.points} == {
        55_000.0,
        65_000.0,
        75_000.0,
    }
    for point in result.points:
        assert point.implied_vol == pytest.approx(0.55, abs=2e-3)
        assert point.vol_unit == "decimal"
        if point.strike < point.forward:
            assert point.option_type == "put"
        else:
            assert point.option_type == "call"


def test_raw_bachelier_iv_uses_normal_volatility_units():
    quotes, expiry_forward = _quotes()
    result = build_raw_iv_points(
        7,
        quotes,
        [expiry_forward],
        "bachelier",
    )

    assert len(result.points) == 3
    assert not result.drops
    assert all(point.implied_vol > 0.0 for point in result.points)
    assert all(
        point.vol_unit == "USD/sqrt(year)"
        for point in result.points
    )