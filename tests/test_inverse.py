import pytest

from deribit.pricing import Black76Model, from_forward_greeks

pytestmark = pytest.mark.pricing

def test_inverse_greeks_obey_quotient_and_ntd_identities():
    index = 65_000.0
    forward = 66_000.0
    strike = 65_000.0
    tau = 0.25
    vol = 0.55
    rate = 0.03
    model = Black76Model()

    cash_price = model.price(
        forward, strike, tau, vol, rate, "call"
    )
    forward_greeks = model.greeks(
        forward, strike, tau, vol, rate, "call"
    )
    inverse = from_forward_greeks(
        cash_price,
        forward_greeks,
        index,
        forward,
    )

    assert inverse.coin_price == pytest.approx(cash_price / index)
    assert inverse.net_transaction_delta == pytest.approx(
        inverse.traditional_spot_delta - inverse.coin_price
    )
    assert inverse.coin_delta == pytest.approx(
        inverse.net_transaction_delta / index
    )
    assert inverse.coin_vega == pytest.approx(
        forward_greeks.vega / index
    )


def test_inverse_greeks_reject_non_positive_conversion_prices():
    model = Black76Model()
    greeks = model.greeks(
        65_000.0, 65_000.0, 0.25, 0.55, 0.03, "call"
    )
    with pytest.raises(ValueError):
        from_forward_greeks(1_000.0, greeks, 0.0, 65_000.0)
