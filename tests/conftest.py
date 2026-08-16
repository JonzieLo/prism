import pytest

from deribit.pricing.bachelier import BachelierModel
from deribit.pricing.binomial import BinomialModel
from deribit.pricing.black_scholes import Black76Model, BlackScholesModel

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


@pytest.fixture
def binom():
    return BinomialModel(steps=200)