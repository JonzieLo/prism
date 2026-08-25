import inspect
import math
import pytest
import deribit.pricing.binomial as binomial_module
from deribit.pricing.bachelier import NoTimeValueError
from deribit.pricing.binomial import BinomialModel
from deribit.pricing.black_scholes import Black76Model

F = 65_000.0
K = 60_000.0
T = 0.5
VOL = 0.55
RATE = 0.04


@pytest.mark.parametrize("cp", ["call", "CALL", "c", "C"])
def test_call_spellings_follow_base(cp):
    model = BinomialModel(100)
    assert model.price(F, K, T, VOL, RATE, cp) == model.price(
        F, K, T, VOL, RATE, "call"
    )


@pytest.mark.parametrize("cp", ["put", "PUT", "p", "P"])
def test_put_spellings_follow_base(cp):
    model = BinomialModel(100)
    assert model.price(F, K, T, VOL, RATE, cp) == model.price(
        F, K, T, VOL, RATE, "put"
    )


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_steps_raise(steps):
    with pytest.raises(ValueError):
        BinomialModel(steps)


@pytest.mark.parametrize(
    "forward,strike,tau,vol",
    [
        (0.0, K, T, VOL),
        (-F, K, T, VOL),
        (F, 0.0, T, VOL),
        (F, -K, T, VOL),
        (F, K, -T, VOL),
        (F, K, T, -VOL),
        (math.inf, K, T, VOL),
    ],
)
def test_invalid_price_inputs_raise(forward, strike, tau, vol):
    with pytest.raises(ValueError):
        BinomialModel().price(forward, strike, tau, vol, RATE, "call")


@pytest.mark.parametrize("cp", ["call", "put"])
def test_expiry_and_zero_volatility_limits(cp):
    model = BinomialModel()
    sign = 1.0 if cp == "call" else -1.0
    intrinsic = max(sign * (F - K), 0.0)
    assert model.price(F, K, 0.0, VOL, RATE, cp) == intrinsic
    assert model.price(F, K, T, 0.0, RATE, cp) == pytest.approx(
        math.exp(-RATE * T) * intrinsic
    )


def test_forward_tree_probability_is_strictly_between_zero_and_one():
    dt = T / 200
    u = math.exp(VOL * math.sqrt(dt))
    p = 1.0 / (1.0 + u)
    assert 0.0 < p < 1.0


def test_monotonicity_and_put_call_parity():
    model = BinomialModel(600)
    call_low = model.price(F, 55_000.0, T, VOL, RATE, "call")
    call_high = model.price(F, 75_000.0, T, VOL, RATE, "call")
    put_low = model.price(F, 55_000.0, T, VOL, RATE, "put")
    put_high = model.price(F, 75_000.0, T, VOL, RATE, "put")
    low_vol = model.price(F, F, T, 0.30, RATE, "call")
    high_vol = model.price(F, F, T, 0.80, RATE, "call")

    assert call_low > call_high
    assert put_low < put_high
    assert low_vol < high_vol

    call = model.price(F, K, T, VOL, RATE, "call")
    put = model.price(F, K, T, VOL, RATE, "put")
    assert call - put == pytest.approx(
        math.exp(-RATE * T) * (F - K), abs=2e-9
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
def test_converges_to_black76(strike, cp, tau, vol, rate):
    lattice = BinomialModel(1_600).price(F, strike, tau, vol, rate, cp)
    closed = Black76Model().price(F, strike, tau, vol, rate, cp)
    assert lattice == pytest.approx(closed, abs=6.0, rel=2e-3)


def test_nonzero_rate_regression_rejects_stock_probability_on_forward_tree():
    model = BinomialModel(1_600)
    closed = Black76Model()
    lattice = model.price(F, F, 1.0, 0.40, 0.12, "call")
    expected = closed.price(F, F, 1.0, 0.40, 0.12, "call")
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
    price = model.price(F, strike, tau, vol, rate, cp)
    recovered = model.implied_vol(price, F, strike, tau, rate, cp)
    repriced = model.price(F, strike, tau, recovered, rate, cp)
    assert recovered == pytest.approx(vol, abs=2e-11, rel=2e-11)
    assert repriced == pytest.approx(price, abs=1e-8, rel=1e-12)


def test_implied_volatility_bounds():
    model = BinomialModel()
    df = math.exp(-RATE * T)
    intrinsic = df * (F - K)
    ceiling = df * F

    with pytest.raises(ValueError):
        model.implied_vol(intrinsic - 1.0, F, K, T, RATE, "call")
    with pytest.raises(NoTimeValueError):
        model.implied_vol(intrinsic, F, K, T, RATE, "call")
    with pytest.raises(ValueError):
        model.implied_vol(ceiling, F, K, T, RATE, "call")
    with pytest.raises(ValueError):
        model.implied_vol(1.0, F, K, 0.0, RATE, "call")


def test_solver_failure_propagates(monkeypatch):
    model = BinomialModel()
    price = model.price(F, K, T, VOL, RATE, "call")

    def fail(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(binomial_module, "brentq", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        model.implied_vol(price, F, K, T, RATE, "call")


@pytest.mark.parametrize("cp", ["call", "put"])
def test_greeks_match_exact_european_benchmark(cp):
    lattice = BinomialModel().greeks(F, K, T, VOL, RATE, cp)
    closed = Black76Model().greeks(F, K, T, VOL, RATE, cp)
    assert lattice == closed
    assert lattice.rho == pytest.approx(
        -T * Black76Model().price(F, K, T, VOL, RATE, cp)
    )