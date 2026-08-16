import math
import pytest
from deribit.pricing.black_scholes import Black76Model, BlackScholesModel
from deribit.pricing.bachelier import BachelierModel, NoTimeValueError

REL_STEP_1ST = 6e-6   # h ~ x * eps^(1/3) for 1st derivatives
REL_STEP_2ND = 1.2e-4 # h ~ x * eps^(1/4) for 2nd derivatives

FORWARD = 65_000.0
TAU = 0.25
RATE = 0.03

NORMAL_VOLS = [13_000.0, 35_750.0, 78_000.0]
LOGNORMAL_VOLS = [0.20, 0.55, 1.20]
STRIKES = [50_000.0, 65_000.0, 80_000.0, 120_000.0]

def central(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def second(f, x, h):
    D = lambda hh: (f(x + hh) - 2.0 * f(x) + f(x - hh)) / (hh * hh) 
    return (4.0 * D(h / 2.0) - D(h)) / 3.0


def assert_close(name, analytic, numeric, rel_tol, floor=1e-12):
    if abs(analytic) < floor:
            assert abs(analytic - numeric) < floor, f"{name}: {analytic!r} vs {numeric!r}"
            return
    err = abs(analytic - numeric) / abs(analytic)
    assert err < rel_tol, f"{name}: analytic={analytic!r} fd={numeric!r} rel_err={err:.3e} > {rel_tol:.1e}"

def otm_leg(strike, reference):
    """The out-of-the-money leg at this strike."""
    return "put" if strike < reference else "call"

@pytest.fixture
def b76():
    return Black76Model()

@pytest.fixture
def bs():
    return BlackScholesModel()

@pytest.fixture
def bach():
    return BachelierModel()

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
    from scipy.stats import norm
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
    """
    The 20-sigma case. At sigma_N = 1500 against F = 65,000 the 50,000 strike is
    twenty standard deviations away; the true time value is order 1e-19 while
    ulp(intrinsic) is 1.8e-12, so the price equals the intrinsic bit for bit.
    Not an arbitrage -- an unrecoverable quote, and it must say so specifically.
    """
    price = bach.price(FORWARD, 50_000.0, TAU, 1500.0, RATE, "call")
    intrinsic = math.exp(-RATE * TAU) * (FORWARD - 50_000.0)
    assert price == intrinsic, "precondition: this quote is at intrinsic bit for bit"
    with pytest.raises(NoTimeValueError):
        bach.implied_vol(price, FORWARD, 50_000.0, TAU, RATE, "call")


def test_sub_intrinsic_is_not_no_time_value(bach):
    """A genuine arbitrage must NOT be reported as an unrecoverable wing."""
    with pytest.raises(ValueError) as excinfo:
        bach.implied_vol(1.0, FORWARD, 50_000.0, TAU, RATE, "call")
    assert not isinstance(excinfo.value, NoTimeValueError)


def test_deep_itm_time_value_survives_the_intrinsic_check(bach):
    """
    Guards the tolerance constant. This quote's time value is 1.9e-06 on a
    74,441 intrinsic -- 133,000 ulp clear of intrinsic, and sigma_N comes back
    to within 6e-08 relative. A tolerance of 1e-10 * intrinsic would be 7.4e-06,
    four times the time value, and would throw this quote away as worthless.
    The check has to be expressed in ulp, not in dollars or in percent.
    """
    strike, vol_n = 140_000.0, 25_000.0
    price = bach.price(FORWARD, strike, TAU, vol_n, RATE, "put")
    intrinsic = math.exp(-RATE * TAU) * (strike - FORWARD)
    time_value = price - intrinsic

    assert time_value > 0.0, "precondition: the quote carries recoverable time value"
    assert time_value < 1e-10 * intrinsic, "precondition: a 1e-10 relative tolerance would reject it"
    assert time_value > 1000.0 * math.ulp(intrinsic), "precondition: but it is many ulp wide"

    recovered = bach.implied_vol(price, FORWARD, strike, TAU, RATE, "put")
    assert abs(recovered - vol_n) / vol_n < 1e-6


def test_implied_vol_rejects_sub_intrinsic(b76):
    forward, strike = 65_000.0, 50_000.0
    with pytest.raises(ValueError):
        b76.implied_vol(1.0, forward, strike, TAU, RATE, "call")