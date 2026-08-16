import math
import pytest
from deribit.pricing.black_scholes import Black76Model, BlackScholesModel

REL_STEP_1ST = 6e-6   # h ~ x * eps^(1/3) for 1st derivatives
REL_STEP_2ND = 1.2e-4 # h ~ x * eps^(1/4) for 2nd derivatives


def central(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def second(f, x, h):
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def assert_close(name, analytic, numeric, rel_tol):
    denom = max(abs(analytic), 1e-30)
    err = abs(analytic - numeric) / denom
    assert err < rel_tol, f"{name}: analytic={analytic!r} fd={numeric!r} rel_err={err:.3e} > {rel_tol:.1e}"

@pytest.fixture
def b76():
    return Black76Model()

@pytest.fixture
def bs():
    return BlackScholesModel()

# Black-76 (Forward-Space) Tests
@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
@pytest.mark.parametrize("cp", ["call", "put"])
@pytest.mark.parametrize("vol", [0.20, 0.55, 1.20])
def test_b76_implied_vol_roundtrip(b76, strike, cp, vol):
    """Price -> implied_vol -> recovers vol in volatility space."""
    forward, tau, rate = 65_000.0, 0.25, 0.03
    price = b76.price(forward, strike, tau, vol, rate, cp)
    recovered = b76.implied_vol(price, forward, strike, tau, rate, cp)
    assert abs(vol - recovered) < 1e-10


@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
def test_b76_put_call_parity(b76, strike):
    """C - P = exp(-r*tau) * (F - K)."""
    forward, tau, rate, vol = 65_000.0, 0.5, 0.04, 0.60
    call = b76.price(forward, strike, tau, vol, rate, "call")
    put = b76.price(forward, strike, tau, vol, rate, "put")
    rhs = (forward - strike) * math.exp(-rate * tau)
    assert abs((call - put) - rhs) < 1e-8


@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
@pytest.mark.parametrize("cp", ["call", "put"])
def test_b76_greeks_vs_fd(b76, strike, cp):
    """Validates Black-76 Forward Greeks against central finite differences."""
    forward, tau, vol, rate = 65_000.0, 0.25, 0.50, 0.03
    g = b76.greeks(forward, strike, tau, vol, rate, cp)

    price = lambda f=forward, v=vol, t=tau, r=rate: b76.price(f, strike, t, v, r, cp)
    vega_of = lambda f=forward, v=vol: b76.greeks(f, strike, tau, v, rate, cp).vega

    hF1 = forward * REL_STEP_1ST
    hF2 = forward * REL_STEP_2ND
    hv1 = vol * REL_STEP_1ST
    ht1 = tau * REL_STEP_1ST
    hr1 = max(rate, 0.01) * REL_STEP_1ST

    assert_close("b76 delta", g.delta, central(lambda f: price(f=f), forward, hF1), 1e-8)
    assert_close("b76 gamma", g.gamma, second(lambda f: price(f=f), forward, hF2), 1e-6)
    assert_close("b76 vega", g.vega, central(lambda v: price(v=v), vol, hv1), 1e-8)
    assert_close("b76 theta", g.theta, -central(lambda t: price(t=t), tau, ht1), 1e-7)
    assert_close("b76 rho", g.rho, central(lambda r: price(r=r), rate, hr1), 1e-7)
    assert_close("b76 vanna", g.vanna, central(lambda f: vega_of(f=f), forward, hF1), 1e-6)
    assert_close("b76 vomma", g.vomma, central(lambda v: vega_of(v=v), vol, hv1), 1e-6)


# Black-Scholes (Spot-Space) Tests
@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
@pytest.mark.parametrize("cp", ["call", "put"])
@pytest.mark.parametrize("vol", [0.20, 0.55, 1.20])
def test_bs_implied_vol_roundtrip(bs, strike, cp, vol):
    """Price -> implied_vol -> recovers vol natively off Spot price S."""
    spot, tau, rate = 65_000.0, 0.25, 0.03
    price = bs.price(spot, strike, tau, vol, rate, cp)
    recovered = bs.implied_vol(price, spot, strike, tau, rate, cp)
    assert abs(vol - recovered) < 1e-10


@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
def test_bs_put_call_parity(bs, strike):
    """C - P = S - K * exp(-r*tau)."""
    spot, tau, rate, vol = 65_000.0, 0.5, 0.04, 0.60
    call = bs.price(spot, strike, tau, vol, rate, "call")
    put = bs.price(spot, strike, tau, vol, rate, "put")
    rhs = spot - strike * math.exp(-rate * tau)
    assert abs((call - put) - rhs) < 1e-8


@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
@pytest.mark.parametrize("cp", ["call", "put"])
def test_bs_greeks_vs_fd(bs, strike, cp):
    """Validates native Black-Scholes Spot Greeks against central finite differences."""
    spot, tau, vol, rate = 65_000.0, 0.25, 0.50, 0.03
    g = bs.greeks(spot, strike, tau, vol, rate, cp)

    price = lambda s=spot, v=vol, t=tau, r=rate: bs.price(s, strike, t, v, r, cp)
    vega_of = lambda s=spot, v=vol: bs.greeks(s, strike, tau, v, rate, cp).vega

    hS1 = spot * REL_STEP_1ST
    hS2 = spot * REL_STEP_2ND
    hv1 = vol * REL_STEP_1ST
    ht1 = tau * REL_STEP_1ST
    hr1 = max(rate, 0.01) * REL_STEP_1ST

    assert_close("bs delta", g.delta, central(lambda s: price(s=s), spot, hS1), 1e-8)
    assert_close("bs gamma", g.gamma, second(lambda s: price(s=s), spot, hS2), 1e-6)
    assert_close("bs vega", g.vega, central(lambda v: price(v=v), vol, hv1), 1e-8)
    assert_close("bs theta", g.theta, -central(lambda t: price(t=t), tau, ht1), 1e-7)
    assert_close("bs rho", g.rho, central(lambda r: price(r=r), rate, hr1), 1e-7)
    assert_close("bs vanna", g.vanna, central(lambda s: vega_of(s=s), spot, hS1), 1e-6)
    assert_close("bs vomma", g.vomma, central(lambda v: vega_of(v=v), vol, hv1), 1e-6)

def test_implied_vol_rejects_sub_intrinsic(b76):
    forward, strike, tau, rate = 65_000.0, 50_000.0, 0.25, 0.03
    with pytest.raises(ValueError):
        b76.implied_vol(1.0, forward, strike, tau, rate, "call")