import math

import pytest

from deribit.pricing.bachelier import NoTimeValueError
from conftest import (
    FORWARD, TAU, RATE, NORMAL_VOLS, STRIKES,
    REL_STEP_1ST, REL_STEP_2ND,
    central, second, assert_close,
)

pytestmark = pytest.mark.pricing

# Bachelier Model Tests
@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
@pytest.mark.parametrize("cp", ["call", "put"])
@pytest.mark.parametrize("vol_n", NORMAL_VOLS)
def test_bachelier_implied_vol_roundtrip(bach, strike, cp, vol_n):
    """Price -> implied_vol -> recovers normal vol in dollar terms."""
    price = bach.price(FORWARD, strike, TAU, vol_n, RATE, cp)
    recovered = bach.implied_vol(price, FORWARD, strike, TAU, RATE, cp)
    assert abs(vol_n - recovered) < 1e-12 * vol_n


@pytest.mark.parametrize("strike", STRIKES)
def test_bachelier_put_call_parity(bach, strike):
    """Bachelier Put-Call Parity: C - P = exp(-r*tau) * (F - K)."""
    tau, rate, vol_n = 0.5, 0.04, 35_750.0
    call = bach.price(FORWARD, strike, tau, vol_n, rate, "call")
    put = bach.price(FORWARD, strike, tau, vol_n, rate, "put")
    rhs = (FORWARD - strike) * math.exp(-rate * tau)
    assert abs((call - put) - rhs) < 1e-12 * max(1.0, abs(rhs))


@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
@pytest.mark.parametrize("cp", ["call", "put"])
def test_bachelier_greeks_vs_fd(bach, strike, cp):
    """Validates analytical Bachelier Greeks against central finite differences."""
    vol_n = 35_750.0
    g = bach.greeks(FORWARD, strike, TAU, vol_n, RATE, cp)
    price = lambda f=FORWARD, v=vol_n, t=TAU, r=RATE: bach.price(f, strike, t, v, r, cp)
    vega_of = lambda f=FORWARD, v=vol_n: bach.greeks(f, strike, TAU, v, RATE, cp).vega

    hF1, hF2 = FORWARD * REL_STEP_1ST, FORWARD * REL_STEP_2ND
    hv1, ht1 = vol_n * REL_STEP_1ST, TAU * REL_STEP_1ST
    hr1 = max(RATE, 0.01) * REL_STEP_1ST

    assert_close("bach delta", g.delta, central(lambda f: price(f=f), FORWARD, hF1), 1e-8)
    assert_close("bach gamma", g.gamma, second(lambda f: price(f=f), FORWARD, hF2), 1e-6)
    assert_close("bach vega", g.vega, central(lambda v: price(v=v), vol_n, hv1), 1e-8)
    assert_close("bach theta", g.theta, -central(lambda t: price(t=t), TAU, ht1), 1e-7)
    assert_close("bach rho", g.rho, central(lambda r: price(r=r), RATE, hr1), 1e-7)
    assert_close("bach vanna", g.vanna, central(lambda f: vega_of(f=f), FORWARD, hF1), 1e-6)
    assert_close("bach vomma", g.vomma, central(lambda v: vega_of(v=v), vol_n, hv1), 1e-6)

def test_bachelier_sigma_n_tracks_forward_times_sigma_ln(bach, b76):
    """
    sigma_N ~ sigma_LN * F at the money.
    """
    for vol_ln, tolerance in [(0.20, 0.01), (0.55, 0.02), (1.20, 0.05)]:
        target = b76.price(FORWARD, FORWARD, TAU, vol_ln, RATE, "call")
        sigma_n = bach.implied_vol(target, FORWARD, FORWARD, TAU, RATE, "call")
        assert abs(sigma_n - vol_ln * FORWARD) / (vol_ln * FORWARD) < tolerance


# Exception and boundary behaviour
def test_far_wing_raises_no_time_value(bach):
    price = bach.price(FORWARD, 50_000.0, TAU, 1500.0, RATE, "call")
    intrinsic = math.exp(-RATE * TAU) * (FORWARD - 50_000.0)
    assert price == intrinsic, "precondition: this quote is at intrinsic bit for bit"
    with pytest.raises(NoTimeValueError):
        bach.implied_vol(price, FORWARD, 50_000.0, TAU, RATE, "call")


def test_sub_intrinsic_is_not_no_time_value(bach):
    with pytest.raises(ValueError) as excinfo:
        bach.implied_vol(1.0, FORWARD, 50_000.0, TAU, RATE, "call")
    assert not isinstance(excinfo.value, NoTimeValueError)


def test_deep_itm_time_value_survives_the_intrinsic_check(bach):
    strike, vol_n = 140_000.0, 25_000.0
    price = bach.price(FORWARD, strike, TAU, vol_n, RATE, "put")
    intrinsic = math.exp(-RATE * TAU) * (strike - FORWARD)
    time_value = price - intrinsic

    assert time_value > 0.0, "precondition: the quote carries recoverable time value"
    assert time_value < 1e-10 * intrinsic, "precondition: a 1e-10 relative tolerance would reject it"
    assert time_value > 1000.0 * math.ulp(intrinsic), "precondition: but it is many ulp wide"

    recovered = bach.implied_vol(price, FORWARD, strike, TAU, RATE, "put")
    assert abs(recovered - vol_n) / vol_n < 1e-6