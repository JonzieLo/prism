import inspect
import math
import pytest
import deribit.pricing.binomial as binomial_module
from deribit.pricing.bachelier import NoTimeValueError
from deribit.pricing.binomial import BinomialModel
from conftest import FORWARD, LOGNORMAL_VOLS, RATE, TAU

STRIKE = 60_000.0
VOL = LOGNORMAL_VOLS[1]


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_steps_raise(steps):
    with pytest.raises(ValueError):
        BinomialModel(steps)


@pytest.mark.parametrize(
    "forward,strike,tau,vol",
    [
        (0.0, STRIKE, TAU, VOL),
        (-FORWARD, STRIKE, TAU, VOL),
        (FORWARD, 0.0, TAU, VOL),
        (FORWARD, -STRIKE, TAU, VOL),
        (FORWARD, STRIKE, -TAU, VOL),
        (FORWARD, STRIKE, TAU, -VOL),
        (math.inf, STRIKE, TAU, VOL),
    ],
)
def test_invalid_price_inputs_raise(binom, forward, strike, tau, vol):
    with pytest.raises(ValueError):
        binom.price(forward, strike, tau, vol, RATE, "call")


@pytest.mark.parametrize("cp", ["call", "put"])
def test_expiry_and_zero_volatility_limits(binom, cp):
    sign = 1.0 if cp == "call" else -1.0
    intrinsic = max(sign * (FORWARD - STRIKE), 0.0)
    assert binom.price(FORWARD, STRIKE, 0.0, VOL, RATE, cp) == intrinsic
    assert binom.price(FORWARD, STRIKE, TAU, 0.0, RATE, cp) == pytest.approx(
        math.exp(-RATE * TAU) * intrinsic
    )


def test_forward_tree_probability_is_strictly_between_zero_and_one():
    dt = TAU / 200
    u = math.exp(VOL * math.sqrt(dt))
    p = 1.0 / (1.0 + u)
    assert 0.0 < p < 1.0


def test_monotonicity_and_put_call_parity():
    model = BinomialModel(600)
    call_low = model.price(FORWARD, 55_000.0, TAU, VOL, RATE, "call")
    call_high = model.price(FORWARD, 75_000.0, TAU, VOL, RATE, "call")
    put_low = model.price(FORWARD, 55_000.0, TAU, VOL, RATE, "put")
    put_high = model.price(FORWARD, 75_000.0, TAU, VOL, RATE, "put")
    low_vol = model.price(FORWARD, FORWARD, TAU, 0.30, RATE, "call")
    high_vol = model.price(FORWARD, FORWARD, TAU, 0.80, RATE, "call")

    assert call_low > call_high
    assert put_low < put_high
    assert low_vol < high_vol

    call = model.price(FORWARD, STRIKE, TAU, VOL, RATE, "call")
    put = model.price(FORWARD, STRIKE, TAU, VOL, RATE, "put")
    assert call - put == pytest.approx(
        math.exp(-RATE * TAU) * (FORWARD - STRIKE), abs=2e-9
    )


def test_implementation_has_no_early_exercise_branch():
    source = inspect.getsource(BinomialModel._tree).lower()
    assert "exercise" not in source
    assert "intrinsic" not in source


@pytest.mark.parametrize(
    "strike,cp,tau,vol,rate",
    [
        (45_000.0, "call", 0.03, 0.15, 0.00),
        (45_000.0, "put", 0.25, 0.55, 0.04),
        (65_000.0, "call", 0.25, 1.20, -0.02),
        (65_000.0, "put", 1.50, 0.55, 0.08),
        (90_000.0, "call", 0.25, 0.55, 0.08),
        (90_000.0, "put", 1.50, 1.20, 0.00),
    ],
)
def test_converges_to_black76(b76, strike, cp, tau, vol, rate):
    lattice = BinomialModel(1_600).price(FORWARD, strike, tau, vol, rate, cp)
    closed = b76.price(FORWARD, strike, tau, vol, rate, cp)
    assert lattice == pytest.approx(closed, abs=6.0, rel=2e-3)


def test_nonzero_rate_regression_rejects_stock_probability_on_forward_tree(b76):
    model = BinomialModel(1_600)
    lattice = model.price(FORWARD, FORWARD, 1.0, 0.40, 0.12, "call")
    expected = b76.price(FORWARD, FORWARD, 1.0, 0.40, 0.12, "call")
    assert lattice == pytest.approx(expected, abs=6.0, rel=1e-3)


@pytest.mark.parametrize(
    "strike,cp,tau,vol,rate",
    [
        (50_000.0, "call", 0.25, 0.20, 0.03),
        (65_000.0, "call", 0.25, 0.55, 0.03),
        (80_000.0, "put", 1.00, 1.20, -0.02),
        (90_000.0, "call", 0.05, 0.80, 0.08),
    ],
)
def test_implied_volatility_round_trip(strike, cp, tau, vol, rate):
    model = BinomialModel(400)
    price = model.price(FORWARD, strike, tau, vol, rate, cp)
    recovered = model.implied_vol(price, FORWARD, strike, tau, rate, cp)
    repriced = model.price(FORWARD, strike, tau, recovered, rate, cp)
    assert recovered == pytest.approx(vol, abs=2e-11, rel=2e-11)
    assert repriced == pytest.approx(price, abs=1e-8, rel=1e-12)


def test_implied_volatility_bounds(binom):
    df = math.exp(-RATE * TAU)
    intrinsic = df * (FORWARD - STRIKE)
    ceiling = df * FORWARD

    with pytest.raises(ValueError):
        binom.implied_vol(intrinsic - 1.0, FORWARD, STRIKE, TAU, RATE, "call")
    with pytest.raises(NoTimeValueError):
        binom.implied_vol(intrinsic, FORWARD, STRIKE, TAU, RATE, "call")
    with pytest.raises(ValueError):
        binom.implied_vol(ceiling, FORWARD, STRIKE, TAU, RATE, "call")
    with pytest.raises(ValueError):
        binom.implied_vol(1.0, FORWARD, STRIKE, 0.0, RATE, "call")


def test_solver_failure_propagates(binom, monkeypatch):
    price = binom.price(FORWARD, STRIKE, TAU, VOL, RATE, "call")

    def fail(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(binomial_module, "brentq", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        binom.implied_vol(price, FORWARD, STRIKE, TAU, RATE, "call")


@pytest.mark.parametrize("cp", ["call", "put"])
def test_lattice_greeks_approach_european_benchmark(b76, cp):
    model = BinomialModel(800)
    lattice = model.greeks(FORWARD, STRIKE, TAU, VOL, RATE, cp)
    closed = b76.greeks(FORWARD, STRIKE, TAU, VOL, RATE, cp)

    assert lattice.delta == pytest.approx(closed.delta, rel=1e-3, abs=1e-5)
    assert lattice.gamma == pytest.approx(closed.gamma, rel=2e-3, abs=1e-8)
    assert lattice.theta == pytest.approx(closed.theta, rel=2e-3, abs=1.0)
    assert lattice.vega == pytest.approx(closed.vega, rel=1e-2, abs=1.0)
    assert lattice.vanna == pytest.approx(closed.vanna, rel=1e-2, abs=1e-5)
    assert lattice.rho == pytest.approx(
        -TAU * model.price(FORWARD, STRIKE, TAU, VOL, RATE, cp)
    )


@pytest.mark.parametrize("cp,sign", [("call", 1), ("put", -1)])
def test_bumped_greeks_match_independent_step_scaled_difference(cp, sign):
    model = BinomialModel(800)
    greeks = model.greeks(FORWARD, STRIKE, TAU, VOL, RATE, cp)
    h = VOL / model.steps
    root = model.price(FORWARD, STRIKE, TAU, VOL, RATE, cp)
    up = model.price(FORWARD, STRIKE, TAU, VOL + h, RATE, cp)
    down = model.price(FORWARD, STRIKE, TAU, VOL - h, RATE, cp)
    delta_up = model._delta(FORWARD, STRIKE, TAU, VOL + h, RATE, sign)
    delta_down = model._delta(FORWARD, STRIKE, TAU, VOL - h, RATE, sign)

    assert greeks.vega == pytest.approx((up - down) / (2.0 * h), rel=1e-5)
    assert greeks.vanna == pytest.approx(
        (delta_up - delta_down) / (2.0 * h), rel=1e-4, abs=1e-8
    )
    assert greeks.vomma == pytest.approx(
        (up - 2.0 * root + down) / (h * h), rel=1e-3, abs=1.0
    )


def test_lattice_greeks_require_two_steps_and_positive_time_and_vol():
    with pytest.raises(ValueError):
        BinomialModel(1).greeks(FORWARD, STRIKE, TAU, VOL, RATE, "call")
    with pytest.raises(ValueError):
        BinomialModel().greeks(FORWARD, STRIKE, 0.0, VOL, RATE, "call")
    with pytest.raises(ValueError):
        BinomialModel().greeks(FORWARD, STRIKE, TAU, 0.0, RATE, "call")