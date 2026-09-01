import math

import pytest
from scipy.stats import norm

from conftest import (
    FORWARD, TAU, RATE, LOGNORMAL_VOLS, STRIKES,
    REL_STEP_1ST, REL_STEP_2ND,
    central, second, assert_close, otm_leg,
)

pytestmark = pytest.mark.pricing

# Black-76 (Forward-Space) Tests
@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("vol", LOGNORMAL_VOLS)
def test_b76_implied_vol_roundtrip_otm(b76, strike, vol):
    """Price -> implied_vol recovers sigma to 1e-10 on the OTM leg."""
    price = b76.price(FORWARD, strike, TAU, vol, RATE, otm_leg(strike, FORWARD))
    recovered = b76.implied_vol(price, FORWARD, strike, TAU, RATE, otm_leg(strike, FORWARD))
    assert abs(vol - recovered) < 1e-10

@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("vol", LOGNORMAL_VOLS)
def test_b76_itm_leg_hits_float64_resolution_floor(b76, strike, vol):
    itm = "call" if strike < FORWARD else "put"
    if strike == FORWARD:
        pytest.skip("no ITM leg at the money")
    price = b76.price(FORWARD, strike, TAU, vol, RATE, itm)
    vega = b76.greeks(FORWARD, strike, TAU, vol, RATE, itm).vega
    resolution = math.ulp(price) / vega
    recovered = b76.implied_vol(price, FORWARD, strike, TAU, RATE, itm)
    assert abs(vol - recovered) < max(1e-10, 4.0 * resolution)

@pytest.mark.parametrize("strike", STRIKES)
def test_b76_put_call_parity(b76, strike):
    """C - P = exp(-r*tau) * (F - K)."""
    tau, rate, vol = 0.5, 0.04, 0.60
    call = b76.price(FORWARD, strike, tau, vol, rate, "call")
    put = b76.price(FORWARD, strike, tau, vol, rate, "put")
    rhs = (FORWARD - strike) * math.exp(-rate * tau)
    assert abs((call - put) - rhs) < 1e-12 * max(1.0, abs(rhs))


@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
@pytest.mark.parametrize("cp", ["call", "put"])
def test_b76_greeks_vs_fd(b76, strike, cp):
    """Validates Black-76 Forward Greeks against central finite differences."""
    vol = 0.50
    g = b76.greeks(FORWARD, strike, TAU, vol, RATE, cp)
    price = lambda f=FORWARD, v=vol, t=TAU, r=RATE: b76.price(f, strike, t, v, r, cp)
    vega_of = lambda f=FORWARD, v=vol: b76.greeks(f, strike, TAU, v, RATE, cp).vega

    hF1, hF2 = FORWARD * REL_STEP_1ST, FORWARD * REL_STEP_2ND
    hv1, ht1 = vol * REL_STEP_1ST, TAU * REL_STEP_1ST
    hr1 = max(RATE, 0.01) * REL_STEP_1ST

    assert_close("b76 delta", g.delta, central(lambda f: price(f=f), FORWARD, hF1), 1e-8)
    assert_close("b76 gamma", g.gamma, second(lambda f: price(f=f), FORWARD, hF2), 1e-6)
    assert_close("b76 vega", g.vega, central(lambda v: price(v=v), vol, hv1), 1e-8)
    assert_close("b76 theta", g.theta, -central(lambda t: price(t=t), TAU, ht1), 1e-7)
    assert_close("b76 rho", g.rho, central(lambda r: price(r=r), RATE, hr1), 1e-7)
    assert_close("b76 vanna", g.vanna, central(lambda f: vega_of(f=f), FORWARD, hF1), 1e-6)
    assert_close("b76 vomma", g.vomma, central(lambda v: vega_of(v=v), vol, hv1), 1e-6)


# Black-Scholes (Spot-Space) Tests
@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("vol", LOGNORMAL_VOLS)
def test_bs_implied_vol_roundtrip(bs, strike, vol):
    """Price -> implied_vol -> recovers vol natively off Spot price S."""
    price = bs.price(FORWARD, strike, TAU, vol, RATE, otm_leg(strike, FORWARD))
    recovered = bs.implied_vol(price, FORWARD, strike, TAU, RATE, otm_leg(strike, FORWARD))
    assert abs(vol - recovered) < 1e-10


@pytest.mark.parametrize("strike", STRIKES)
def test_bs_put_call_parity(bs, strike):
    """C - P = S - K * exp(-r*tau)."""
    tau, rate, vol = 0.5, 0.04, 0.60
    call = bs.price(FORWARD, strike, tau, vol, rate, "call")
    put = bs.price(FORWARD, strike, tau, vol, rate, "put")
    rhs = FORWARD - strike * math.exp(-rate * tau)
    assert abs((call - put) - rhs) < 1e-12 * max(1.0, abs(rhs))


@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
@pytest.mark.parametrize("cp", ["call", "put"])
def test_bs_greeks_vs_fd(bs, strike, cp):
    """Validates native Black-Scholes Spot Greeks against central finite differences."""
    vol = 0.50
    g = bs.greeks(FORWARD, strike, TAU, vol, RATE, cp)
    price = lambda s=FORWARD, v=vol, t=TAU, r=RATE: bs.price(s, strike, t, v, r, cp)
    vega_of = lambda s=FORWARD, v=vol: bs.greeks(s, strike, TAU, v, RATE, cp).vega

    hS1, hS2 = FORWARD * REL_STEP_1ST, FORWARD * REL_STEP_2ND
    hv1, ht1 = vol * REL_STEP_1ST, TAU * REL_STEP_1ST
    hr1 = max(RATE, 0.01) * REL_STEP_1ST

    assert_close("bs delta", g.delta, central(lambda s: price(s=s), FORWARD, hS1), 1e-8)
    assert_close("bs gamma", g.gamma, second(lambda s: price(s=s), FORWARD, hS2), 1e-6)
    assert_close("bs vega", g.vega, central(lambda v: price(v=v), vol, hv1), 1e-8)
    assert_close("bs theta", g.theta, -central(lambda t: price(t=t), TAU, ht1), 1e-7)
    assert_close("bs rho", g.rho, central(lambda r: price(r=r), RATE, hr1), 1e-7)
    assert_close("bs vanna", g.vanna, central(lambda s: vega_of(s=s), FORWARD, hS1), 1e-6)
    assert_close("bs vomma", g.vomma, central(lambda v: vega_of(v=v), vol, hv1), 1e-6)

@pytest.mark.parametrize("cp", ["call", "put"])
def test_bs_greeks_are_spot_greeks(bs, cp):
    strike, vol = 65_000.0, 0.50
    g = bs.greeks(FORWARD, strike, TAU, vol, RATE, cp)
    d1 = (math.log(FORWARD / strike) + (RATE + 0.5 * vol * vol) * TAU) / (vol * math.sqrt(TAU))
    d2 = d1 - vol * math.sqrt(TAU)
    df = math.exp(-RATE * TAU)

    if cp == "call":
        assert g.delta == pytest.approx(norm.cdf(d1), rel=1e-12)
        assert g.rho == pytest.approx(strike * TAU * df * norm.cdf(d2), rel=1e-10)
        assert g.rho > 0.0, "a BS call is long rates; a Black-76 call is short them"
    else:
        assert g.delta == pytest.approx(norm.cdf(d1) - 1.0, rel=1e-12)
        assert g.rho == pytest.approx(-strike * TAU * df * norm.cdf(-d2), rel=1e-10)
        assert g.rho < 0.0, "a BS put is short rates"


def test_implied_vol_rejects_sub_intrinsic(b76):
    forward, strike = 65_000.0, 50_000.0
    with pytest.raises(ValueError):
        b76.implied_vol(1.0, forward, strike, TAU, RATE, "call")