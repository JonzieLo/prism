import pytest
from deribit.chain import OptionQuote, FutureQuote, futures_from_snapshot
from deribit.forwards import (
    OptionPair,
    PairingIssue,
    inverse_forward_mid,
    synthetic_buy_forward,
    synthetic_sell_forward,
)

pytestmark = pytest.mark.forwards

def test_futures_from_snapshot_maps_market_fields():
    ...
    

def make_dummy_quote(
    source_row_id: int = 1,
    instrument_name: str = "BTC-25DEC26-75000-C",
    option_type: str = "call",
    strike: float = 75000.0,
    expiration_timestamp: int = 1798185600000,
    underlying_index: str = "BTC-25DEC26",
    settlement_currency: str = "BTC",
    contract_size: float = 1.0,
    bid_coin: float | None = 0.10,
    ask_coin: float | None = 0.10,
) -> OptionQuote:
    """Helper to quickly construct valid OptionQuote objects for testing."""
    return OptionQuote(
        source_row_id=source_row_id,
        instrument_name=instrument_name,
        option_type=option_type,
        strike=strike,
        expiration_timestamp=expiration_timestamp,
        underlying_index=underlying_index,
        settlement_currency=settlement_currency,
        contract_size=contract_size,
        index_price=75000.0,
        api_forward=75000.0,
        api_rate=0.0,
        tau=0.25,
        bid_coin=bid_coin,
        ask_coin=ask_coin,
        bid_amount=10.0,
        ask_amount=10.0,
        mark_coin=0.10,
        deribit_mark_iv=0.55,
        open_interest=100.0,
        volume=50.0,
        last_coin=0.10,
    )


#  --- Future Normalization ---
def test_futures_from_snapshot_maps_market_fields():
    snapshot = {
        "futures": {
            "payload": [
                {
                    "instrument_name": "BTC-25DEC26",
                    "bid_price": 80100.0,
                    "ask_price": 80110.0,
                    "mark_price": 80105.0,
                    "last": 80102.0,
                    # bid_amount
                    # ask_amount
                    "open_interest": 1500.0,
                    "contract_size": 1.0,
                }
            ]
        }
    }

    result = futures_from_snapshot(snapshot)

    assert "BTC-25DEC26" in result

    future = result["BTC-25DEC26"]
    assert future.instrument_name == "BTC-25DEC26"
    assert future.bid == 80100.0
    assert future.ask == 80110.0
    assert future.mark == 80105.0
    assert future.last == 80102.0

    # missing fields are None
    assert future.bid_amount is None
    assert future.ask_amount is None


def test_futures_from_snapshot_raises_on_duplicates():
    snapshot = {
        "futures": {
            "payload": [
                {"instrument_name": "BTC-25DEC26", "mark_price": 80000.0},
                {"instrument_name": "BTC-25DEC26", "mark_price": 80100.0},
            ]
        }
    }

    with pytest.raises(ValueError, match="Duplicate future row: BTC-25DEC26"):
        futures_from_snapshot(snapshot)


# --- Midpoint Recovery ---
def test_inverse_forward_mid_recovery():
    call = make_dummy_quote(option_type="call", strike=75000.0, bid_coin=0.10, ask_coin=0.10)
    put = make_dummy_quote(option_type="put", strike=75000.0, bid_coin=0.0375, ask_coin=0.0375)
    
    pair = OptionPair(
        expiration_timestamp=1798185600000,
        underlying_index="BTC-25DEC26",
        strike=75000.0,
        call=call,
        put=put,
    )

    # Assert inverse_forward_mid == 80000.0
    assert inverse_forward_mid(pair) == pytest.approx(80000.0)


# --- Exact Bid-Ask Endpoints ---
def test_exact_bid_ask_endpoints():
    # Call bid/ask = 0.0990 / 0.1010
    call = make_dummy_quote(option_type="call", strike=75000.0, bid_coin=0.0990, ask_coin=0.1010)
    # Put bid/ask = 0.0365 / 0.0385
    put = make_dummy_quote(option_type="put", strike=75000.0, bid_coin=0.0365, ask_coin=0.0385)

    pair = OptionPair(
        expiration_timestamp=1798185600000,
        underlying_index="BTC-25DEC26",
        strike=75000.0,
        call=call,
        put=put,
    )

    sell_fw = synthetic_sell_forward(pair)
    buy_fw = synthetic_buy_forward(pair)
    mid_fw = inverse_forward_mid(pair)

    assert sell_fw == pytest.approx(79829.69664715274)
    assert buy_fw == pytest.approx(80171.03153393907)

    assert sell_fw <= mid_fw <= buy_fw


# --- Independent Side Eligibility ---
def test_independent_side_eligibility():
    # Zero call bid -> sell side None, buy side not None
    call_zero_bid = make_dummy_quote(option_type="call", bid_coin=0.0, ask_coin=0.1010)
    put_normal = make_dummy_quote(option_type="put", bid_coin=0.0365, ask_coin=0.0385)
    pair1 = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call_zero_bid, put_normal)

    assert synthetic_sell_forward(pair1) is None
    assert synthetic_buy_forward(pair1) is not None

    # Zero put bid --> buy side None, sell side not None
    call_normal = make_dummy_quote(option_type="call", bid_coin=0.0990, ask_coin=0.1010)
    put_zero_bid = make_dummy_quote(option_type="put", bid_coin=0.0, ask_coin=0.0385)
    pair2 = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call_normal, put_zero_bid)

    assert synthetic_buy_forward(pair2) is None
    assert synthetic_sell_forward(pair2) is not None

    # Missing midpoint
    # (e.g. call ask exists and put bid exists for buy side, but call bid is None so mid_coin is None)
    call_no_mid = make_dummy_quote(option_type="call", bid_coin=None, ask_coin=0.1010)
    pair3 = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call_no_mid, put_normal)

    assert inverse_forward_mid(pair3) is None
    assert synthetic_buy_forward(pair3) is not None  # Buy side still valid!