import math
import pytest
from deribit.pricing.black_scholes import Black76Model
from deribit.pricing.merton import MertonModel, MertonJumpParams
pytestmark = pytest.mark.merton
FORWARD = 65_000.0
TAU = 0.25
RATE = 0.03
VOL = 0.50

def test_merton_zero_intensity_matches_black76():
    """When lambda = 0, Merton must equal Black-76"""
    b76 = Black76Model()
    merton = MertonModel(MertonJumpParams(intensity=0.0))
    for strike in (50_000.0, 65_000.0, 80_000.0):
        for cp in ("call", "put"):
            b76_px = b76.price(FORWARD, strike, TAU, VOL, RATE, cp)
            mjd_px = merton.price(FORWARD, strike, TAU, VOL, RATE, cp)
            assert abs(b76_px - mjd_px) < 1e-12

def test_merton_put_call_parity():
    """C - P = exp(-r*tau) * (F - K)."""
    merton = MertonModel(MertonJumpParams(intensity=3.0, jump_mean=-0.10, jump_std=0.20))
    strike = 70_000.0
    call = merton.price(FORWARD, strike, TAU, VOL, RATE, "call")
    put = merton.price(FORWARD, strike, TAU, VOL, RATE, "put")
    expected = (FORWARD - strike) * math.exp(-RATE * TAU)
    assert abs((call - put) - expected) < 1e-10

def test_merton_implied_vol_roundtrip():
    """Price -> implied_vol recovers original diffusion sigma."""
    merton = MertonModel(MertonJumpParams(intensity=1.5, jump_mean=-0.05, jump_std=0.15))
    strike = 65_000.0
    px = merton.price(FORWARD, strike, TAU, VOL, RATE, "call")
    recovered = merton.implied_vol(px, FORWARD, strike, TAU, RATE, "call")
    assert abs(recovered - VOL) < 1e-6