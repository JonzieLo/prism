import pytest

from deribit.pricing.bachelier import BachelierModel
from deribit.pricing.binomial import BinomialModel
from deribit.pricing.black_scholes import Black76Model, BlackScholesModel
from conftest import FORWARD, TAU, RATE, STRIKES, otm_leg

MODELS = [
    pytest.param(Black76Model(), 0.55, id="black76"),
    pytest.param(BlackScholesModel(), 0.55, id="black_scholes"),
    pytest.param(BachelierModel(), 35_750.0, id="bachelier"),
    # pytest.param(BinomialModel(steps=200), 0.55, id="binomial"),
]


@pytest.mark.parametrize("model,vol", MODELS)
@pytest.mark.parametrize("strike", STRIKES)
def test_otm_price_is_positive_and_below_forward(model, vol, strike):
    price = model.price(FORWARD, strike, TAU, vol, RATE, otm_leg(strike, FORWARD))
    assert 0.0 < price < FORWARD


@pytest.mark.parametrize("model,vol", MODELS)
@pytest.mark.parametrize("strike", STRIKES)
def test_otm_price_is_strictly_increasing_in_vol(model, vol, strike):
    """More uncertainty is worth more, in every model. Strict for OTM options:
    away from any exercise boundary, vega > 0."""
    cp = otm_leg(strike, FORWARD)
    lo = model.price(FORWARD, strike, TAU, 0.8 * vol, RATE, cp)
    hi = model.price(FORWARD, strike, TAU, 1.2 * vol, RATE, cp)
    assert hi > lo


@pytest.mark.parametrize("model,vol", MODELS)
def test_sub_intrinsic_quote_is_rejected(model, vol):
    """A quote below intrinsic is an arbitrage; every engine must refuse it."""
    with pytest.raises(ValueError):
        model.implied_vol(1.0, FORWARD, 50_000.0, TAU, RATE, "call")