import pytest
import math
from scipy.stats import norm
from deribit.pricing import Black76Model, BlackScholesModel, from_forward_greeks

pytestmark = pytest.mark.pricing

def test_inverse_greeks_obey_quotient_and_ntd_identities():
    index = 65_000.0
    strike = 65_000.0
    tau = 0.25
    vol = 0.55
    rate = 0.03
    forward = index * math.exp(rate * tau)
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
        tau,
        rate
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
        from_forward_greeks(1_000.0, greeks, 0.0, 65_000.0, 0.25, 0.03)


@pytest.mark.parametrize("cp", ["call", "put"])
@pytest.mark.parametrize("strike", [50_000.0, 65_000.0, 80_000.0])
def test_inverse_greeks_match_paper_spot_convention(cp, strike):
    """Map Black-76 Greeks to Table A.1's spot-held-fixed convention."""
    spot = 65_000.0
    tau = 0.40
    vol = 0.55
    rate = 0.03
    forward = spot * math.exp(rate * tau)
    b76 = Black76Model()
    bs = BlackScholesModel()

    cash_price = b76.price(forward, strike, tau, vol, rate, cp)
    forward_greeks = b76.greeks(
        forward, strike, tau, vol, rate, cp
    )
    spot_greeks = bs.greeks(spot, strike, tau, vol, rate, cp)
    inverse = from_forward_greeks(
        cash_price,
        forward_greeks,
        spot,
        forward,
        tau,
        rate,
    )

    assert cash_price == pytest.approx(
        bs.price(spot, strike, tau, vol, rate, cp), rel=1e-13
    )
    assert inverse.traditional_spot_delta == pytest.approx(
        spot_greeks.delta, rel=1e-13
    )
    assert inverse.coin_delta == pytest.approx(
        (spot * spot_greeks.delta - cash_price) / spot**2,
        rel=1e-13,
        abs=1e-16,
    )
    assert inverse.coin_gamma == pytest.approx(
        spot_greeks.gamma / spot
        - 2.0 * spot_greeks.delta / spot**2
        + 2.0 * cash_price / spot**3,
        rel=1e-12,
        abs=1e-18,
    )
    assert inverse.coin_vega == pytest.approx(
        spot_greeks.vega / spot, rel=1e-13
    )
    assert inverse.coin_theta == pytest.approx(
        spot_greeks.theta / spot, rel=1e-13
    )
    assert inverse.coin_rho == pytest.approx(
        spot_greeks.rho / spot, rel=1e-13, abs=1e-16
    )
    assert inverse.coin_vanna == pytest.approx(
        spot_greeks.vanna / spot - spot_greeks.vega / spot**2,
        rel=1e-12,
        abs=1e-16,
    )
    assert inverse.coin_vomma == pytest.approx(
        spot_greeks.vomma / spot, rel=1e-12, abs=1e-16
    )

    sqrt_tau = math.sqrt(tau)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * vol * vol) * tau
    ) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    expected_ntd = (
        strike / forward * norm.cdf(d2)
        if cp == "call"
        else -strike / forward * norm.cdf(-d2)
    )
    assert inverse.net_transaction_delta == pytest.approx(
        expected_ntd, rel=1e-13
    )


@pytest.mark.parametrize("cp", ["call", "put"])
def test_coin_greeks_match_direct_finite_differences(cp):
    """Differentiate c(S, sigma, r, tau) = BS(S, ...)/S directly."""
    spot = 65_000.0
    strike = 60_000.0
    tau = 0.40
    vol = 0.55
    rate = 0.03
    forward = spot * math.exp(rate * tau)
    b76 = Black76Model()
    bs = BlackScholesModel()

    inverse = from_forward_greeks(
        b76.price(forward, strike, tau, vol, rate, cp),
        b76.greeks(forward, strike, tau, vol, rate, cp),
        spot,
        forward,
        tau,
        rate,
    )

    def coin_value(
        s: float = spot,
        sigma: float = vol,
        t: float = tau,
        r: float = rate,
    ) -> float:
        return bs.price(s, strike, t, sigma, r, cp) / s

    def coin_vega(s: float = spot, sigma: float = vol) -> float:
        return bs.greeks(s, strike, tau, sigma, rate, cp).vega / s

    hs = 1.0
    hv = 1e-5
    ht = 1e-5
    hr = 1e-5

    fd_delta = (
        coin_value(s=spot + hs) - coin_value(s=spot - hs)
    ) / (2.0 * hs)
    fd_gamma = (
        coin_value(s=spot + hs)
        - 2.0 * coin_value()
        + coin_value(s=spot - hs)
    ) / hs**2
    fd_vega = (
        coin_value(sigma=vol + hv) - coin_value(sigma=vol - hv)
    ) / (2.0 * hv)
    fd_theta = -(
        coin_value(t=tau + ht) - coin_value(t=tau - ht)
    ) / (2.0 * ht)
    fd_rho = (
        coin_value(r=rate + hr) - coin_value(r=rate - hr)
    ) / (2.0 * hr)
    fd_vanna = (
        coin_vega(s=spot + hs) - coin_vega(s=spot - hs)
    ) / (2.0 * hs)
    fd_vomma = (
        coin_vega(sigma=vol + hv) - coin_vega(sigma=vol - hv)
    ) / (2.0 * hv)

    assert inverse.coin_delta == pytest.approx(
        fd_delta, rel=2e-9, abs=1e-14
    )
    assert inverse.coin_gamma == pytest.approx(
        fd_gamma, rel=2e-5, abs=1e-16
    )
    assert inverse.coin_vega == pytest.approx(
        fd_vega, rel=2e-9, abs=1e-10
    )
    assert inverse.coin_theta == pytest.approx(
        fd_theta, rel=2e-9, abs=1e-10
    )
    assert inverse.coin_rho == pytest.approx(
        fd_rho, rel=2e-9, abs=1e-10
    )
    assert inverse.coin_vanna == pytest.approx(
        fd_vanna, rel=2e-8, abs=1e-12
    )
    assert inverse.coin_vomma == pytest.approx(
        fd_vomma, rel=2e-8, abs=1e-10
    )