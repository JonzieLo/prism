import pytest
from deribit.chain import OptionQuote, FutureQuote, futures_from_snapshot, option_chain_from_snapshot
from deribit.forwards import *
from deribit.hygiene import evaluate_and_partition_pairs

pytestmark = pytest.mark.forwards


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


def test_one_call_and_one_put_create_one_pair():
    call = make_dummy_quote(source_row_id=1, option_type="call")
    put = make_dummy_quote(source_row_id=2, option_type="put", instrument_name="BTC-25DEC26-75000-P")
    
    pairs, issues = pair_calls_and_puts([call, put])
    
    assert len(pairs) == 1
    assert len(issues) == 0
    assert pairs[0].call == call
    assert pairs[0].put == put


def test_missing_call_is_reported():
    put = make_dummy_quote(source_row_id=1, option_type="put")
    
    pairs, issues = pair_calls_and_puts([put])
    
    assert len(pairs) == 0
    assert len(issues) == 1
    assert issues[0].reason == "missing_call"


def test_missing_put_is_reported():
    call = make_dummy_quote(source_row_id=1, option_type="call")
    
    pairs, issues = pair_calls_and_puts([call])
    
    assert len(pairs) == 0
    assert len(issues) == 1
    assert issues[0].reason == "missing_put"


def test_duplicate_calls_are_reported():
    call1 = make_dummy_quote(source_row_id=1, option_type="call")
    call2 = make_dummy_quote(source_row_id=2, option_type="call")
    put = make_dummy_quote(source_row_id=3, option_type="put")
    
    pairs, issues = pair_calls_and_puts([call1, call2, put])
    
    assert len(pairs) == 0
    reasons = [issue.reason for issue in issues]
    assert "duplicate_call" in reasons


def test_duplicate_puts_are_reported():
    call = make_dummy_quote(source_row_id=1, option_type="call")
    put1 = make_dummy_quote(source_row_id=2, option_type="put")
    put2 = make_dummy_quote(source_row_id=3, option_type="put")
    
    pairs, issues = pair_calls_and_puts([call, put1, put2])
    
    assert len(pairs) == 0
    reasons = [issue.reason for issue in issues]
    assert "duplicate_put" in reasons


def test_different_expiry_does_not_pair():
    call = make_dummy_quote(source_row_id=1, option_type="call", expiration_timestamp=1798185600000)
    put = make_dummy_quote(source_row_id=2, option_type="put", expiration_timestamp=1800000000000)
    
    pairs, issues = pair_calls_and_puts([call, put])
    
    assert len(pairs) == 0
    assert len(issues) == 2  # One missing_put, one missing_call


def test_different_underlying_future_does_not_pair():
    call = make_dummy_quote(source_row_id=1, option_type="call", underlying_index="BTC-25DEC26")
    put = make_dummy_quote(source_row_id=2, option_type="put", underlying_index="BTC-26MAR27")
    
    pairs, issues = pair_calls_and_puts([call, put])
    
    assert len(pairs) == 0
    assert len(issues) == 2


def test_settlement_currency_mismatch_does_not_pair():
    call = make_dummy_quote(source_row_id=1, option_type="call", settlement_currency="BTC")
    put = make_dummy_quote(source_row_id=2, option_type="put", settlement_currency="USDC")
    
    pairs, issues = pair_calls_and_puts([call, put])
    
    assert len(pairs) == 0
    assert len(issues) == 1
    assert issues[0].reason == "settlement_currency_mismatch"


def test_contract_size_mismatch_does_not_pair():
    call = make_dummy_quote(source_row_id=1, option_type="call", contract_size=1.0)
    put = make_dummy_quote(source_row_id=2, option_type="put", contract_size=10.0)
    
    pairs, issues = pair_calls_and_puts([call, put])
    
    assert len(pairs) == 0
    assert len(issues) == 1
    assert issues[0].reason == "contract_size_mismatch"

## Parity Point Tests
def make_point(
    strike: float,
    forward_mid: float | None = 80000.0,
    synthetic_buy: float | None = 80100.0,
    synthetic_sell: float | None = 79900.0,
    expiration: int = 1798185600000,
    underlying: str = "BTC-25DEC26",
) -> ParityPoint:
    return ParityPoint(
        expiration_timestamp=expiration,
        underlying_index=underlying,
        strike=strike,
        forward_mid=forward_mid,
        synthetic_buy_forward=synthetic_buy,
        synthetic_sell_forward=synthetic_sell,
        synthetic_buy_max_size=10.0,
        synthetic_sell_max_size=10.0,
    )


def test_exact_median_for_odd_point_count():
    points = [
        make_point(70000.0, forward_mid=79000.0),
        make_point(75000.0, forward_mid=80000.0),
        make_point(80000.0, forward_mid=81000.0),
    ]

    agg = aggregate_expiry_forward(points, points, points)

    assert agg.implied_forward == 80000.0
    assert agg.pair_count == 3


def test_exact_median_for_even_point_count():
    points = [
        make_point(70000.0, forward_mid=79000.0),
        make_point(75000.0, forward_mid=80000.0),
        make_point(80000.0, forward_mid=81000.0),
        make_point(85000.0, forward_mid=82000.0),
    ]

    agg = aggregate_expiry_forward(points, points, points)

    assert agg.implied_forward == 80500.0


def test_exact_mad_and_iqr():
    points = [
        make_point(70000.0, forward_mid=78000.0),
        make_point(75000.0, forward_mid=80000.0),
        make_point(80000.0, forward_mid=82000.0),
    ]

    agg = aggregate_expiry_forward(points, points, points)

    # Median = 80000. Absolute diffs: [2000, 0, 2000] -> Median MAD = 2000
    assert agg.dispersion_mad == 2000.0
    assert agg.dispersion_iqr == 2000.0


def test_best_buy_selects_minimum_eligible_synthetic_buy_forward():
    points = [
        make_point(70000.0, synthetic_buy=80500.0),
        make_point(75000.0, synthetic_buy=80100.0),  # Best (lowest cost to buy)
        make_point(80000.0, synthetic_buy=80300.0),
    ]

    agg = aggregate_expiry_forward(points, points, points)

    assert agg.best_synthetic_buy == 80100.0
    assert agg.best_synthetic_buy_strike == 75000.0


def test_best_sell_selects_maximum_eligible_synthetic_sell_forward():
    points = [
        make_point(70000.0, synthetic_sell=79500.0),
        make_point(75000.0, synthetic_sell=79900.0),  # Best (highest price to sell)
        make_point(80000.0, synthetic_sell=79700.0),
    ]

    agg = aggregate_expiry_forward(points, points, points)

    assert agg.best_synthetic_sell == 79900.0
    assert agg.best_synthetic_sell_strike == 75000.0


def test_ineligible_extreme_quote_cannot_become_best_buy_or_sell():
    diagnostic_points = [make_point(75000.0, forward_mid=80000.0)]
    buy_points = [make_point(75000.0, synthetic_buy=80200.0)]
    sell_points = [make_point(75000.0, synthetic_sell=79800.0)]

    agg = aggregate_expiry_forward(diagnostic_points, buy_points, sell_points)

    assert agg.best_synthetic_buy == 80200.0
    assert agg.best_synthetic_sell == 79800.0


def test_empty_diagnostic_set_raises():
    with pytest.raises(ValueError, match="Expiry has no diagnostic-eligible parity points"):
        aggregate_expiry_forward([], [], [])


def test_mixed_expiry_raises():
    points = [
        make_point(75000.0, expiration=1798185600000),
        make_point(75000.0, expiration=1800000000000),
    ]

    with pytest.raises(ValueError, match="Cannot aggregate multiple expiries or underlyings"):
        aggregate_expiry_forward(points, points, points)


def test_mixed_underlying_future_raises():
    points = [
        make_point(75000.0, underlying="BTC-25DEC26"),
        make_point(75000.0, underlying="BTC-26MAR27"),
    ]

    with pytest.raises(ValueError, match="Cannot aggregate multiple expiries or underlyings"):
        aggregate_expiry_forward(points, points, points)


## ExpiryForward Tests, Implied vs Traded Futures
def make_expiry_forward(
    implied_forward: float = 80000.0,
    best_buy: float | None = 80100.0,
    best_sell: float | None = 79900.0,
) -> ExpiryForward:
    return ExpiryForward(
        expiration_timestamp=1798185600000,
        underlying_index="BTC-25DEC26",
        implied_forward=implied_forward,
        dispersion_mad=100.0,
        dispersion_iqr=150.0,
        pair_count=5,
        best_synthetic_buy=best_buy,
        best_synthetic_buy_strike=75000.0,
        best_synthetic_sell=best_sell,
        best_synthetic_sell_strike=75000.0,
    )


def make_future(
    bid: float | None = 79950.0,
    ask: float | None = 80050.0,
    mark: float | None = 80000.0,
) -> FutureQuote:
    return FutureQuote(
        instrument_name="BTC-25DEC26",
        bid=bid,
        ask=ask,
        bid_amount=10.0,
        ask_amount=10.0,
        mark=mark,
        last=80000.0,
        open_interest=1000.0,
        contract_size=1.0,
    )


def test_correct_usd_basis():
    ef = make_expiry_forward(implied_forward=80200.0)
    future = make_future(mark=80000.0)

    comparison = compare_with_future(ef, future)

    assert comparison.basis_usd == pytest.approx(200.0)


def test_correct_basis_points():
    ef = make_expiry_forward(implied_forward=80800.0)
    future = make_future(mark=80000.0)

    comparison = compare_with_future(ef, future)

    # 10,000 * (80800 / 80000 - 1) = 100 bps
    assert comparison.basis_bps == pytest.approx(100.0)


def test_missing_future_has_explicit_status():
    ef = make_expiry_forward()

    comparison = compare_with_future(ef, None)

    assert comparison.status == BasisStatus.MISSING_FUTURE.value
    assert comparison.basis_usd is None
    assert comparison.basis_bps is None


def test_invalid_mark_leaves_midpoint_basis_unavailable():
    ef = make_expiry_forward()
    future = make_future(mark=None)

    comparison = compare_with_future(ef, future)

    assert comparison.basis_usd is None
    assert comparison.basis_bps is None
    assert comparison.status == BasisStatus.INVALID_FUTURE_MARK.value


def test_valid_future_bid_can_still_produce_price_cross_when_mark_unavailable():
    ef = make_expiry_forward(best_buy=80000.0)
    future = make_future(bid=80100.0, mark=None)

    comparison = compare_with_future(ef, future)

    assert comparison.top_of_book_cross_direction == "buy_synthetic_sell_future"
    assert comparison.status == BasisStatus.CROSSED_FUTURE.value


def test_equality_produces_no_cross():
    ef = make_expiry_forward(best_buy=80100.0, best_sell=79900.0)
    future = make_future(bid=80100.0, ask=79900.0)

    comparison = compare_with_future(ef, future)

    assert comparison.top_of_book_cross_direction is None
    assert comparison.status == BasisStatus.OK.value


def test_buy_synthetic_sell_future_direction_is_correct():
    # Future bid > best synthetic buy
    ef = make_expiry_forward(best_buy=80000.0)
    future = make_future(bid=80100.0, ask=80200.0)

    comparison = compare_with_future(ef, future)

    assert comparison.top_of_book_cross_direction == "buy_synthetic_sell_future"
    assert comparison.status == BasisStatus.CROSSED_FUTURE.value


def test_buy_future_sell_synthetic_direction_is_correct():
    # Best synthetic sell > future ask
    ef = make_expiry_forward(best_sell=80300.0)
    future = make_future(bid=80100.0, ask=80200.0)

    comparison = compare_with_future(ef, future)

    assert comparison.top_of_book_cross_direction == "sell_synthetic_buy_future"
    assert comparison.status == BasisStatus.CROSSED_FUTURE.value


def test_both_directions_crossing_is_reported_as_invalid_data():
    # Future bid > best synthetic buy AND best synthetic sell > future ask
    ef = make_expiry_forward(best_buy=80000.0, best_sell=80500.0)
    future = make_future(bid=80100.0, ask=80200.0)

    comparison = compare_with_future(ef, future)

    assert comparison.top_of_book_cross_direction == "both_directions_cross"
    assert comparison.status == BasisStatus.BOTH_DIRECTIONS_CROSS.value


def test_forward_derivation_without_api_forward_or_mark_iv():
    snapshot = {
        "index": {
            "received_at_ns": 1700000000000000000,
            "payload": {"index_price": 75000.0},
        },
        "instruments": {
            "received_at_ns": 1700000000000000000,
            "payload": [
                {
                    "kind": "option",
                    "instrument_name": "BTC-25DEC26-75000-C",
                    "option_type": "call",
                    "strike": 75000,
                    "expiration_timestamp": 1798185600000,
                    "settlement_currency": "BTC",
                    "contract_size": 1.0,
                },
                {
                    "kind": "option",
                    "instrument_name": "BTC-25DEC26-75000-P",
                    "option_type": "put",
                    "strike": 75000,
                    "expiration_timestamp": 1798185600000,
                    "settlement_currency": "BTC",
                    "contract_size": 1.0,
                },
            ],
        },
        "options": {
            "received_at_ns": 1700000000000000000,
            "payload": [
                {
                    "instrument_name": "BTC-25DEC26-75000-C",
                    "underlying_index": "BTC-25DEC26",
                    "underlying_price": None,  # Missing API forward
                    "mark_iv": None,           # Missing Mark IV
                    "bid_price": 0.10,
                    "ask_price": 0.10,
                },
                {
                    "instrument_name": "BTC-25DEC26-75000-P",
                    "underlying_index": "BTC-25DEC26",
                    "underlying_price": None,  # Missing API forward
                    "mark_iv": None,           # Missing Mark IV
                    "bid_price": 0.0375,
                    "ask_price": 0.0375,
                },
            ],
        },
    }

    chain = option_chain_from_snapshot(snapshot)
    assert len(chain) == 2
    pairs, issues = pair_calls_and_puts(chain)
    assert len(pairs) == 1
    diag, buy, sell, evaluated = evaluate_and_partition_pairs(pairs)
    assert len(diag) == 1
    expiry_fw = aggregate_expiry_forward(diag, buy, sell)
    assert expiry_fw.implied_forward == pytest.approx(80000.0)


def test_input_quote_reconciliation():
    c1 = make_dummy_quote(source_row_id=10, option_type="call", strike=70000.0)
    p1 = make_dummy_quote(source_row_id=11, option_type="put", strike=70000.0)
    c2_dup1 = make_dummy_quote(source_row_id=12, option_type="call", strike=75000.0)
    c2_dup2 = make_dummy_quote(source_row_id=13, option_type="call", strike=75000.0)
    p2 = make_dummy_quote(source_row_id=14, option_type="put", strike=75000.0)

    input_quotes = [c1, p1, c2_dup1, c2_dup2, p2]
    pairs, issues = pair_calls_and_puts(input_quotes)

    input_ids = {q.source_row_id for q in input_quotes}
    
    paired_ids = set()
    for pair in pairs:
        paired_ids.add(pair.call.source_row_id)
        paired_ids.add(pair.put.source_row_id)

    issue_ids = set()
    for issue in issues:
        issue_ids.update(issue.source_row_ids)

    assert input_ids == paired_ids | issue_ids
    assert len(paired_ids & issue_ids) == 0


def test_one_extreme_wing_does_not_materially_move_median_forward():
    normal_points = [
        make_point(70000.0, forward_mid=79950.0),
        make_point(75000.0, forward_mid=80000.0),
        make_point(80000.0, forward_mid=80050.0),
    ]

    clean_agg = aggregate_expiry_forward(normal_points, normal_points, normal_points)

    outlier_points = normal_points + [make_point(120000.0, forward_mid=150000.0)]
    outlier_agg = aggregate_expiry_forward(outlier_points, outlier_points, outlier_points)

    # Median is stable (moves from 80000 to 80025)
    assert abs(outlier_agg.implied_forward - clean_agg.implied_forward) <= 50.0
    assert outlier_agg.dispersion_mad >= clean_agg.dispersion_mad


def test_compare_with_future_invalid_futures_book():
    ef = make_expiry_forward()
    # Crossed futures book: bid > ask
    future_crossed = make_future(bid=80200.0, ask=80100.0)

    comparison = compare_with_future(ef, future_crossed)
    assert comparison.status == BasisStatus.INVALID_FUTURE_BOOK.value
    assert comparison.top_of_book_cross_direction is None


def test_compare_with_future_non_finite_or_non_positive():
    ef = make_expiry_forward()
    # Non-positive mark
    future_zero_mark = make_future(mark=0.0)
    assert compare_with_future(ef, future_zero_mark).basis_usd is None

    # Non-finite price
    future_nan = make_future(bid=float("nan"))
    assert compare_with_future(ef, future_nan).status == BasisStatus.INVALID_FUTURE_BOOK.value