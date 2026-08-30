import pytest
from deribit.forward_curve import build_forward_curve
from deribit.forwards import BasisStatus


pytestmark = pytest.mark.forwards


def make_snapshot(
    *,
    call_bid: float = 0.099,
    call_ask: float = 0.101,
    put_bid: float = 0.0365,
    put_ask: float = 0.0385,
    include_future: bool = True,
) -> dict:
    received_at_ns = 1_700_000_000_000_000_000
    expiry = 1_798_185_600_000
    call_name = "BTC-25DEC26-75000-C"
    put_name = "BTC-25DEC26-75000-P"
    instruments = [
        {
            "kind": "option",
            "instrument_name": call_name,
            "option_type": "call",
            "strike": 75_000.0,
            "expiration_timestamp": expiry,
            "settlement_currency": "BTC",
            "contract_size": 1.0,
        },
        {
            "kind": "option",
            "instrument_name": put_name,
            "option_type": "put",
            "strike": 75_000.0,
            "expiration_timestamp": expiry,
            "settlement_currency": "BTC",
            "contract_size": 1.0,
        },
    ]
    options = [
        {
            "instrument_name": call_name,
            "underlying_index": "BTC-25DEC26",
            "underlying_price": None,
            "mark_iv": None,
            "bid_price": call_bid,
            "ask_price": call_ask,
            "bid_amount": 5.0,
            "ask_amount": 6.0,
        },
        {
            "instrument_name": put_name,
            "underlying_index": "BTC-25DEC26",
            "underlying_price": None,
            "mark_iv": None,
            "bid_price": put_bid,
            "ask_price": put_ask,
            "bid_amount": 7.0,
            "ask_amount": 8.0,
        },
    ]
    futures = []
    if include_future:
        futures.append(
            {
                "instrument_name": "BTC-25DEC26",
                "bid_price": 79_990.0,
                "ask_price": 80_010.0,
                "mark_price": 80_000.0,
                "bid_amount": 10.0,
                "ask_amount": 10.0,
            }
        )
    return {
        "index": {
            "received_at_ns": received_at_ns,
            "payload": {"index_price": 75_000.0},
        },
        "instruments": {
            "received_at_ns": received_at_ns,
            "payload": instruments,
        },
        "options": {
            "received_at_ns": received_at_ns,
            "payload": options,
        },
        "futures": {
            "received_at_ns": received_at_ns,
            "payload": futures,
        },
    }


def test_build_forward_curve_runs_without_api_forward_or_mark_iv():
    result = build_forward_curve(make_snapshot())

    assert result.raw_option_count == 2
    assert len(result.quotes) == 2
    assert result.paired_quote_count == 2
    assert result.pairing_issue_quote_count == 0
    assert result.diagnostic_pair_count == 1
    assert result.expiry_forwards[0].implied_forward == pytest.approx(80_000.0)
    assert result.comparisons[0].basis_bps == pytest.approx(0.0)
    assert result.comparisons[0].status == BasisStatus.OK.value

    counts = {item.reason: item.pair_count for item in result.filter_counts}
    assert counts["missing_api_forward"] == 1
    assert counts["missing_mark_iv"] == 1