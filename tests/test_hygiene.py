import pytest
from deribit.chain import OptionQuote
from deribit.forwards import OptionPair
from deribit.hygiene import IssueCode, Use, evaluate_pair

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