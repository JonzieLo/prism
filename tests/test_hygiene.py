import pytest
from deribit.chain import OptionQuote
from deribit.forwards import OptionPair, inverse_forward_mid, synthetic_buy_forward, synthetic_sell_forward
from deribit.hygiene import IssueCode, Use, evaluate_pair, evaluate_and_partition_pairs

pytestmark = pytest.mark.forwards

def make_dummy_quote(
    option_type: str = "call",
    bid_coin: float | None = 0.10,
    ask_coin: float | None = 0.10,
    api_forward: float | None = 75000.0,
    deribit_mark_iv: float | None = 0.55,
) -> OptionQuote:
    """Helper to quickly construct valid OptionQuote objects for testing."""
    return OptionQuote(
        source_row_id=1,
        instrument_name=f"BTC-25DEC26-75000-{'C' if option_type == 'call' else 'P'}",
        option_type=option_type,
        strike=75000.0,
        expiration_timestamp=1798185600000,
        underlying_index="BTC-25DEC26",
        settlement_currency="BTC",
        contract_size=1.0,
        index_price=75000.0,
        api_forward=api_forward,
        api_rate=0.0,
        tau=0.25,
        bid_coin=bid_coin,
        ask_coin=ask_coin,
        bid_amount=10.0,
        ask_amount=10.0,
        mark_coin=0.10,
        deribit_mark_iv=deribit_mark_iv,
        open_interest=100.0,
        volume=50.0,
        last_coin=0.10,
    )


def test_zero_call_bid_blocks_midpoint_and_synthetic_sell_but_not_buy():
    call = make_dummy_quote(option_type="call", bid_coin=0.0, ask_coin=0.10)
    put = make_dummy_quote(option_type="put", bid_coin=0.03, ask_coin=0.03)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    assert not evaluated.diagnostic_eligible
    assert not evaluated.synthetic_sell_eligible
    assert evaluated.synthetic_buy_eligible


def test_zero_put_bid_blocks_midpoint_and_synthetic_buy_but_not_sell():
    call = make_dummy_quote(option_type="call", bid_coin=0.10, ask_coin=0.10)
    put = make_dummy_quote(option_type="put", bid_coin=0.0, ask_coin=0.03)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    assert not evaluated.diagnostic_eligible
    assert evaluated.synthetic_sell_eligible
    assert not evaluated.synthetic_buy_eligible


def test_missing_call_ask_blocks_midpoint_and_synthetic_buy_but_not_sell():
    call = make_dummy_quote(option_type="call", bid_coin=0.10, ask_coin=None)
    put = make_dummy_quote(option_type="put", bid_coin=0.03, ask_coin=0.03)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    assert not evaluated.diagnostic_eligible
    assert not evaluated.synthetic_buy_eligible
    assert evaluated.synthetic_sell_eligible


def test_missing_put_ask_blocks_midpoint_and_synthetic_sell_but_not_buy():
    call = make_dummy_quote(option_type="call", bid_coin=0.10, ask_coin=0.10)
    put = make_dummy_quote(option_type="put", bid_coin=0.03, ask_coin=None)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    assert not evaluated.diagnostic_eligible
    assert not evaluated.synthetic_sell_eligible
    assert evaluated.synthetic_buy_eligible


def test_crossed_call_book_blocks_all_three_uses():
    call = make_dummy_quote(option_type="call", bid_coin=0.12, ask_coin=0.10)
    put = make_dummy_quote(option_type="put", bid_coin=0.03, ask_coin=0.03)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    assert not evaluated.diagnostic_eligible
    assert not evaluated.synthetic_buy_eligible
    assert not evaluated.synthetic_sell_eligible


def test_crossed_put_book_blocks_all_three_uses():
    call = make_dummy_quote(option_type="call", bid_coin=0.10, ask_coin=0.10)
    put = make_dummy_quote(option_type="put", bid_coin=0.05, ask_coin=0.03)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    assert not evaluated.diagnostic_eligible
    assert not evaluated.synthetic_buy_eligible
    assert not evaluated.synthetic_sell_eligible


def test_missing_api_forward_emits_non_blocking_issue():
    call = make_dummy_quote(option_type="call", api_forward=None)
    put = make_dummy_quote(option_type="put")
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    codes = [issue.code for issue in evaluated.issues]
    assert IssueCode.MISSING_API_FORWARD in codes
    assert evaluated.diagnostic_eligible
    assert evaluated.synthetic_buy_eligible
    assert evaluated.synthetic_sell_eligible


def test_missing_mark_iv_emits_non_blocking_issue():
    call = make_dummy_quote(option_type="call", deribit_mark_iv=None)
    put = make_dummy_quote(option_type="put")
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    codes = [issue.code for issue in evaluated.issues]
    assert IssueCode.MISSING_MARK_IV in codes
    assert evaluated.diagnostic_eligible
    assert evaluated.synthetic_buy_eligible
    assert evaluated.synthetic_sell_eligible


def test_one_pair_can_have_multiple_issues():
    call = make_dummy_quote(option_type="call", bid_coin=0.0, api_forward=None)
    put = make_dummy_quote(option_type="put", ask_coin=None)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    assert len(evaluated.issues) >= 3


def test_evaluate_and_partition_pairs_filters_correctly():
    call_zero_put_bid = make_dummy_quote("call", bid_coin=0.10, ask_coin=0.10)
    put_zero_bid = make_dummy_quote("put", bid_coin=0.0, ask_coin=0.03)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call_zero_put_bid, put_zero_bid)

    diag, buy, sell, evaluated = evaluate_and_partition_pairs([pair])
    evaluated_point = evaluated[0].point

    assert evaluated_point not in diag
    assert evaluated_point not in buy
    assert evaluated_point in sell
    assert len(evaluated) == 1


def test_spread_wider_than_midpoint_blocks_mid_only():
    # Spread (0.19 - 0.01 = 0.18) > Midpoint (0.10)
    call = make_dummy_quote("call", bid_coin=0.01, ask_coin=0.19)
    put = make_dummy_quote("put", bid_coin=0.03, ask_coin=0.03)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)

    codes = [issue.code for issue in evaluated.issues]
    assert IssueCode.SPREAD_WIDER_THAN_MID in codes
    assert not evaluated.diagnostic_eligible
    assert evaluated.synthetic_buy_eligible
    assert evaluated.synthetic_sell_eligible


def test_invalid_denominators_separately():
    call = make_dummy_quote("call", bid_coin=1.01, ask_coin=1.02)
    put = make_dummy_quote("put", bid_coin=0.001, ask_coin=0.002)
    pair = OptionPair(1798185600000, "BTC-25DEC26", 75000.0, call, put)

    evaluated = evaluate_pair(pair)
    # (1.0 - call_mid + put_mid) <= 0

    evaluated = evaluate_pair(pair)

    codes = [issue.code for issue in evaluated.issues]
    assert IssueCode.INVALID_MID_DENOMINATOR in codes
    assert IssueCode.INVALID_BUY_DENOMINATOR in codes
    assert IssueCode.INVALID_SELL_DENOMINATOR in codes
    assert not evaluated.diagnostic_eligible
    assert not evaluated.synthetic_buy_eligible
    assert not evaluated.synthetic_sell_eligible