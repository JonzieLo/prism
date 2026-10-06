import pytest
import math
from deribit.chain import OptionQuote
from deribit.forwards import ExpiryForward
from deribit.pricing import Black76Model
from deribit.surface.pipeline import build_surface_observations

def _quote(
    source_row_id: int,
    option_type: str,
    strike: float,
    index: float,
    forward: float,
    tau: float,
    rate: float,
) -> OptionQuote:
    model = Black76Model()
    mid_usd = model.price(
        forward, strike, tau, 0.55, rate, option_type
    )
    mid_coin = mid_usd / index
    suffix = "C" if option_type == "call" else "P"
    return OptionQuote(
        source_row_id=source_row_id,
        instrument_name=f"BTC-TEST-{strike:.0f}-{suffix}",
        option_type=option_type,
        strike=strike,
        expiration_timestamp=1_800_000_000_000,
        underlying_index="BTC-TEST",
        settlement_currency="BTC",
        contract_size=1.0,
        index_price=index,
        tau=tau,
        bid_coin=0.95 * mid_coin,
        ask_coin=1.05 * mid_coin,
        bid_amount=1.0,
        ask_amount=1.0,
        mark_coin=mid_coin,
        deribit_mark_iv=None,
        open_interest=100.0,
        volume=10.0,
        last_coin=mid_coin,
    )


def test_canonical_otm_leg_selection_and_independence():
    index = 65_000.0
    forward = 66_000.0
    tau = 0.40
    rate = math.log(forward / index) / tau
    strikes = [55_000.0, 65_000.0, 75_000.0]
    quotes = [
        _quote(
            source_row_id,
            option_type,
            strike,
            index,
            forward,
            tau,
            rate,
        )
        for source_row_id, (strike, option_type) in enumerate(
            (
                (strike, option_type)
                for strike in strikes
                for option_type in ("call", "put")
            ),
            start=1,
        )
    ]
    expiry_forward = ExpiryForward(
        expiration_timestamp=1_800_000_000_000,
        underlying_index="BTC-TEST",
        implied_forward=forward,
        dispersion_mad=0.0,
        dispersion_iqr=0.0,
        pair_count=len(strikes),
        best_synthetic_buy=None,
        best_synthetic_buy_strike=None,
        best_synthetic_sell=None,
        best_synthetic_sell_strike=None,
    )

    res = build_surface_observations(
        snapshot_id=1,
        quotes=quotes,
        expiry_forwards=[expiry_forward],
    )
    keys = [(obs.expiration_timestamp, obs.strike) for obs in res.observations]
    assert res.snapshot_id == 1
    assert len(keys) == len(set(keys))
    assert len(keys) == len(strikes)
    assert not res.exclusions

    for obs in res.observations:
        assert obs.snapshot_id == res.snapshot_id
        assert obs.candidate_key[0] == res.snapshot_id
        if obs.strike < obs.forward:
            assert obs.option_type == "put"
        elif obs.strike > obs.forward:
            assert obs.option_type == "call"

        assert obs.mid_iv > 0.0
        assert obs.total_variance == pytest.approx(obs.mid_iv ** 2 * obs.tau)


def test_surface_observations_reject_invalid_snapshot_id():
    with pytest.raises(ValueError, match="invalid snapshot_id"):
        build_surface_observations(
            snapshot_id=0,
            quotes=[],
            expiry_forwards=[],
        )