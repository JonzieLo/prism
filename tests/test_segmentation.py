import math
import pytest
from deribit.chain import OptionQuote
from deribit.forwards import ExpiryForward, ParityPoint
from deribit.hygiene import EvaluatedPair
from deribit.segmentation import *

pytestmark = pytest.mark.segmentation

def test_invalid_inputs():
    with pytest.raises(ValueError, match="finite and positive"):
        log_moneyness(0.0, 70000.0)
    with pytest.raises(ValueError, match="finite and positive"):
        log_moneyness(70000.0, -100.0)
    with pytest.raises(ValueError, match="finite and positive"):
        log_moneyness(math.nan, 70000.0)

    with pytest.raises(ValueError, match="must be finite"):
        moneyness_band(math.nan)

def test_relative_spread_validation():
    def make_quote(bid, ask):
        return OptionQuote(
            source_row_id=1,
            instrument_name="BTC-TEST",
            option_type="call",
            strike=100000.0,
            expiration_timestamp=1000,
            underlying_index="BTC",
            settlement_currency="BTC",
            contract_size=1.0,
            index_price=70000.0,
            tau=0.1,
            bid_coin=bid,
            ask_coin=ask,
            bid_amount=1.0,
            ask_amount=1.0,
            mark_coin=0.01,
            deribit_mark_iv=0.6,
            open_interest=10.0,
            volume=5.0,
            last_coin=0.01,
        )

    # Valid spread: (0.02 - 0.01) / 0.015 = 0.66666...
    q_valid = make_quote(0.01, 0.02)
    assert relative_spread(q_valid) == pytest.approx(2 / 3)

    assert relative_spread(make_quote(None, 0.02)) is None
    assert relative_spread(make_quote(0.0, 0.02)) is None
    assert relative_spread(make_quote(0.03, 0.02)) is None

def test_observed_pair_sum():
    assert observed_pair_sum(10.0, 20.0) == 30.0
    assert observed_pair_sum(None, 20.0) is None
    assert observed_pair_sum(10.0, math.nan) is None

def test_signed_residuals_calculation():
    fwd = ExpiryForward(
        expiration_timestamp=1000,
        underlying_index="BTC",
        implied_forward=70000.0,
        dispersion_mad=0.0,
        dispersion_iqr=0.0,
        pair_count=3,
        best_synthetic_buy=70000.0,
        best_synthetic_buy_strike=70000.0,
        best_synthetic_sell=70000.0,
        best_synthetic_sell_strike=70000.0,
    )

    def make_eval_pair(strike, fwd_mid):
        call = OptionQuote(
            source_row_id=1,
            instrument_name="C",
            option_type="call",
            strike=strike,
            expiration_timestamp=1000,
            underlying_index="BTC",
            settlement_currency="BTC",
            contract_size=1.0,
            index_price=70000.0,
            tau=0.1,
            bid_coin=0.01,
            ask_coin=0.02,
            bid_amount=1.0,
            ask_amount=1.0,
            mark_coin=0.01,
            deribit_mark_iv=0.6,
            open_interest=10.0,
            volume=5.0,
            last_coin=0.01,
        )
        put = OptionQuote(
            source_row_id=2,
            instrument_name="P",
            option_type="put",
            strike=strike,
            expiration_timestamp=1000,
            underlying_index="BTC",
            settlement_currency="BTC",
            contract_size=1.0,
            index_price=70000.0,
            tau=0.1,
            bid_coin=0.01,
            ask_coin=0.02,
            bid_amount=1.0,
            ask_amount=1.0,
            mark_coin=0.01,
            deribit_mark_iv=0.6,
            open_interest=10.0,
            volume=5.0,
            last_coin=0.01,
        )
        point = ParityPoint(1000, "BTC", strike, fwd_mid, None, None, None, None)
        return EvaluatedPair(
            pair=type(
                "Pair",
                (),
                {
                    "underlying_index": "BTC",
                    "expiration_timestamp": 1000,
                    "strike": strike,
                    "call": call,
                    "put": put,
                },
            )(),
            point=point,
            issues=(),
        )

    pairs = [
        make_eval_pair(70000.0, 69900.0),  # e = -100 USD
        make_eval_pair(70000.0, 70000.0),  # e =    0 USD
        make_eval_pair(70000.0, 70100.0),  # e = +100 USD
    ]

    metrics = build_moneyness_segmentation(pairs, fwd)
    near_fwd = next(m for m in metrics if m.band == MoneynessBand.NEAR_FORWARD)

    assert near_fwd.median_abs_internal_residual_usd == pytest.approx(100.0)
    assert near_fwd.internal_residual_mad_usd == pytest.approx(100.0)