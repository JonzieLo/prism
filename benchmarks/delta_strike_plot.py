import argparse
import asyncio
import gzip
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Mapping

from deribit.forward_curve import build_forward_curve
from deribit.chain import (
    OptionQuote,
    option_chain_from_snapshot,
    select_expiry,
)
from deribit.config import SnapshotUniversalConfig
from deribit.pricing import (
    BachelierModel,
    BinomialModel,
    Black76Model,
    from_forward_greeks,
)
from deribit.hygiene import Use, evaluate_leg
from deribit.store import SnapshotStore
from deribit.ws_client import DeribitWSClient


@dataclass(frozen=True)
class DeltaRow:
    quote: OptionQuote
    lognormal_vol: float
    normal_vol: float
    traditional_delta: float
    black76_delta: float
    binomial_delta: float
    bachelier_delta: float
    net_transaction_delta: float
    diagnostic: bool
    issue_codes: tuple[str,...]
    bid_delta: float | None = None
    ask_delta: float | None = None


async def fetch_snapshot(
    store: SnapshotStore,
    currency: str,
    testnet: bool,
) -> int:
    config = SnapshotUniversalConfig(currency=currency)
    client = DeribitWSClient(testnet=testnet)
    try:
        snapshot = await client.fetch_snapshot_data(config)
    finally:
        await client.close()
    return store.save_snapshot(currency, snapshot)


def build_delta_rows(
    quotes: list[OptionQuote],
    steps: int,
    forward_by_expiry: Mapping[tuple[str, int], float],
) -> tuple[list[DeltaRow], list[tuple[str, str]]]:
    black76 = Black76Model()
    bachelier = BachelierModel()
    binomial = BinomialModel(steps)
    rows: list[DeltaRow] = []
    dropped: list[tuple[str, str]] = []

    for quote in quotes:
        mid_usd = quote.mid_usd
        if mid_usd is None:
            dropped.append((quote.instrument_name, "no two-sided midpoint"))
            continue
        
        quote_issues = evaluate_leg(quote, quote.option_type)
        diagnostic = not any(
            Use.DIAGNOSTIC_MID in issue.blocks for issue in quote_issues
        )
        issue_codes = tuple(issue.code.value for issue in quote_issues)

        try:
            key = (
                quote.underlying_index,
                quote.expiration_timestamp,
            )

            forward = forward_by_expiry.get(key)
            if forward is None:
                dropped.append((quote.instrument_name,"No parity-derived forward for expiry",))
                continue

            rate = math.log(forward / quote.index_price) / quote.tau
            lognormal_vol = black76.implied_vol(mid_usd, forward, quote.strike, quote.tau, rate, quote.option_type)
            normal_vol = bachelier.implied_vol(mid_usd, forward, quote.strike, quote.tau, rate, quote.option_type)
            black76_greeks = black76.greeks(forward, quote.strike, quote.tau, lognormal_vol, rate, quote.option_type)
            binomial_greeks = binomial.greeks(forward, quote.strike, quote.tau, lognormal_vol, rate, quote.option_type)
            bachelier_greeks = bachelier.greeks(forward, quote.strike, quote.tau, normal_vol, rate, quote.option_type)
            cash_price = black76.price(forward, quote.strike, quote.tau, lognormal_vol, rate, quote.option_type)
            inverse = from_forward_greeks(cash_price, black76_greeks, quote.index_price, forward)

            bid_delta = None
            ask_delta = None
            if (
                quote.bid_coin is not None 
                and quote.ask_coin is not None
                and quote.bid_coin > 0.0 
                and quote.ask_coin > 0.0
            ):
                try:
                    bid_usd = quote.index_price * quote.bid_coin
                    ask_usd = quote.index_price * quote.ask_coin

                    bid_iv = black76.implied_vol(bid_usd, forward, quote.strike, quote.tau, rate, quote.option_type)
                    ask_iv = black76.implied_vol(ask_usd, forward, quote.strike, quote.tau, rate, quote.option_type)

                    bid_greeks = black76.greeks(forward, quote.strike, quote.tau, bid_iv, rate, quote.option_type)
                    ask_greeks = black76.greeks(forward, quote.strike, quote.tau, ask_iv, rate, quote.option_type)

                    bid_cash = black76.price(forward, quote.strike, quote.tau, bid_iv, rate, quote.option_type)
                    ask_cash = black76.price(forward, quote.strike, quote.tau, ask_iv, rate, quote.option_type)

                    bid_delta = from_forward_greeks(bid_cash, bid_greeks, quote.index_price, forward).traditional_spot_delta
                    ask_delta = from_forward_greeks(ask_cash, ask_greeks, quote.index_price, forward).traditional_spot_delta
                except ValueError:
                    pass

        except ValueError as error:
            dropped.append((quote.instrument_name, str(error)))
            continue

        rows.append(
            DeltaRow(
                quote=quote,
                lognormal_vol=lognormal_vol,
                normal_vol=normal_vol,
                traditional_delta=inverse.traditional_spot_delta,
                black76_delta=black76_greeks.delta,
                binomial_delta=binomial_greeks.delta,
                bachelier_delta=bachelier_greeks.delta,
                net_transaction_delta=inverse.net_transaction_delta,
                diagnostic=diagnostic,
                issue_codes=issue_codes,
                bid_delta=bid_delta,
                ask_delta=ask_delta,
            )
        )

    return rows, dropped


def plot_delta_rows(
    rows: list[DeltaRow],
    output: Path,
    snapshot_id: int | str,
    dropped_count: int,
    steps: int,
    model_forward: float,
) -> None:
    if not rows:
        raise ValueError("no valid option rows remain after IV inversion")

    eligible_rows = [r for r in rows if r.diagnostic]
    excluded_rows = [r for r in rows if not r.diagnostic]

    el_strikes = np.array([r.quote.strike for r in eligible_rows])
    el_traditional = np.array([r.traditional_delta for r in eligible_rows])
    el_black76 = np.array([r.black76_delta for r in eligible_rows])
    el_binomial = np.array([r.binomial_delta for r in eligible_rows])
    el_bachelier = np.array([r.bachelier_delta for r in eligible_rows])
    el_ntd = np.array([r.net_transaction_delta for r in eligible_rows])

    ex_strikes = np.array([r.quote.strike for r in excluded_rows])
    ex_traditional = np.array([r.traditional_delta for r in excluded_rows])
    ex_ntd = np.array([r.net_transaction_delta for r in excluded_rows])

    abs_diffs = [
        (abs(row.lognormal_vol - row.quote.deribit_mark_iv), row.quote.instrument_name)
        for row in rows
        if row.quote.deribit_mark_iv is not None
    ]

    if abs_diffs:
        diff_values = [d[0] for d in abs_diffs]
        max_diff, max_inst = max(abs_diffs, key=lambda x: x[0])
        p95_diff = float(np.percentile(diff_values, 95))
        median_diff = float(np.median(diff_values))
        iv_summary = (
            f"median |own − mark IV|={median_diff:.4f} | "
            f"p95={p95_diff:.4f} | max={max_diff:.4f} ({max_inst})"
        )
    else:
        iv_summary = "median |own − mark IV|=n/a"

    first = rows[0].quote

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "figure.facecolor": "#F7F6F2",
            "axes.facecolor": "#F7F6F2",
            "axes.edgecolor": "#D4D1CA",
            "text.color": "#28251D",
            "axes.labelcolor": "#28251D",
            "xtick.color": "#28251D",
            "ytick.color": "#28251D",
        }
    ):
        figure, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True,
            layout="constrained",
        )

        top.plot(
            el_strikes,
            el_traditional,
            color="#20808D",
            marker="o",
            markersize=3,
            lw=2.2,
            label="Spot-equivalent traditional delta",
        )
        top.plot(
            el_strikes,
            el_black76,
            color="#A84B2F",
            marker="s",
            markersize=3,
            lw=1.8,
            ls="--",
            label="Black–76 forward delta",
        )
        top.plot(
            el_strikes,
            el_binomial,
            color="#1B474D",
            lw=1.4,
            label=f"CRR forward delta ({steps} steps)",
        )
        top.plot(
            el_strikes,
            el_bachelier,
            color="#7A39BB",
            lw=1.8,
            ls="-.",
            label="Price-matched Bachelier delta",
        )

        for r in rows:
            if r.bid_delta is not None and r.ask_delta is not None:
                top.vlines(
                    r.quote.strike,
                    r.bid_delta,
                    r.ask_delta,
                    color="#BAB9B4",
                    linewidth=1.2,
                    alpha=0.75,
                )

        if len(ex_strikes) > 0:
            top.scatter(
                ex_strikes,
                ex_traditional,
                color="#A13544",
                marker="x",
                s=40,
                zorder=4,
                label="Excluded leg midpoint",
            )

        top.set_ylabel("Delta")
        top.set_title(
            "Traditional deltas use market-mid implied volatility\n"
            "Deribit mark IV is retained only as a cross-check",
            loc="left",
        )
        top.legend(frameon=False, ncol=2, loc="best")

        bottom.plot(
            el_strikes,
            el_ntd,
            color="#DA7101",
            marker="o",
            markersize=3,
            lw=2.4,
            label="Inverse Net Transaction Delta",
        )

        if len(ex_strikes) > 0:
            bottom.scatter(
                ex_strikes,
                ex_ntd,
                color="#A13544",
                marker="x",
                s=40,
                zorder=4,
                label="Excluded leg midpoint",
            )

        bottom.axhline(0.0, color="#7A7974", lw=0.8)
        bottom.set_xlabel("Strike K (USD)")
        bottom.set_ylabel("Base-coin exposure")
        bottom.set_title(
            "Inverse delta subtracts the coin option premium\n"
            r"$\mathrm{NTD}=\Delta_{\mathrm{traditional}}-c"
            r"=X\,\partial(C/X)/\partial X$",
            loc="left",
        )
        bottom.legend(frameon=False, loc="best")

        for axis in (top, bottom):
            axis.axvline(
                first.index_price, 
                color="#7A7974", 
                ls=":", 
                lw=1.2, 
                label=f"Index X (${first.index_price:,.0f})"
            )
            axis.axvline(
                model_forward, 
                color="#28251D", 
                ls="--", 
                lw=1.2, 
                label=f"Forward F (${model_forward:,.0f})"
            )
            axis.grid(axis="y", color="#D4D1CA", lw=0.8, alpha=0.7)
            axis.grid(axis="x", visible=False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        expiry = datetime.fromtimestamp(
            first.expiration_timestamp / 1000.0,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M UTC")
        figure.suptitle(
            f"{first.option_type.title()} delta by strike from Deribit Snapshot",
            fontsize=17,
            fontweight="bold",
            x=0.01,
            ha="left",
        )

        figure.text(
            0.01,
            -0.02,
            (
                f"Expiry {expiry} | X={first.index_price:,.2f} | "
                f"F={model_forward:,.2f} | kept={len(rows)} | "
                f"excluded={len(excluded_rows)} | dropped={dropped_count} | "
                f"{iv_summary}"
            ),
            fontsize=8.5,
            color="#7A7974",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            dpi=180,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model and inverse deltas from a Deribit snapshot."
    )
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--snapshot-id", type=int)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--expiry", type=int)
    parser.add_argument("--option-type", choices=("call", "put"), default="call")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", default="delta_vs_strike.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SnapshotStore(args.db)

    if args.fetch:
        snapshot_id = asyncio.run(
            fetch_snapshot(store, args.currency, args.testnet)
        )
        snapshot = store.load_snapshot(snapshot_id)
    elif args.snapshot_id is not None:
        snapshot_id = args.snapshot_id
        snapshot = store.load_snapshot(snapshot_id)
    else:
        snapshot_id = store.latest_snapshot_id(args.currency)
        if snapshot_id is None:
            raise SystemExit(
                "No stored snapshot; use --fetch or --snapshot-id"
            )
        snapshot = store.load_snapshot(snapshot_id)
    if snapshot is None:
        raise SystemExit(f"Nnapshot {snapshot_id} does not exist")

    curve_result = build_forward_curve(snapshot)
    forward_by_expiry = {
        (
            expiry_forward.underlying_index,
            expiry_forward.expiration_timestamp,
        ): expiry_forward.implied_forward
        for expiry_forward in curve_result.expiry_forwards
    }
    chain = option_chain_from_snapshot(snapshot)
    selected = select_expiry(chain, args.option_type, args.expiry)

    selected_key = (
        selected[0].underlying_index,
        selected[0].expiration_timestamp,
    )
    if selected_key not in forward_by_expiry:
        raise SystemExit(
            "Selected expiry has no diagnostic-eligible options-implied forward."
        )
    rows, dropped = build_delta_rows(selected, args.steps, forward_by_expiry,)
    model_forward = forward_by_expiry[selected_key]
    sorted_quotes = sorted(selected, key=lambda quote: quote.strike)
    for lower, higher in zip(sorted_quotes, sorted_quotes[1:]):
        if (
            lower.mid_coin is not None
            and higher.mid_coin is not None
            and higher.mid_coin > lower.mid_coin
        ):
            print(
                f"midpoint monotonicity warning: {lower.instrument_name} ({lower.mid_coin:.4f}) < "
                f"{higher.instrument_name} ({higher.mid_coin:.4f})"
            )
        if (
            lower.ask_coin is not None
            and higher.bid_coin is not None
            and higher.bid_coin > lower.ask_coin
        ):
            print(
                f"executable call monotonicity violation: Sell {higher.instrument_name} @ {higher.bid_coin:.4f} > "
                f"Buy {lower.instrument_name} @ {lower.ask_coin:.4f}"
            )
    print("\n--- STRIKE-LEVEL DIAGNOSTIC ($200k - $240k) ---")
    print(
        f"{'Instrument':<24} {'Strike':<8} {'Bid':<8} {'Ask':<8} {'Mid':<8} "
        f"{'RelSpread':<10} {'OwnIV':<8} {'MarkIV':<8} {'Delta':<8} {'NTD':<8} {'OI':<8} {'Vol':<8}"
    )
    for row in rows:
        quote = row.quote
        if 200_000 <= quote.strike <= 240_000:
            spread = (
                quote.ask_coin - quote.bid_coin
                if quote.bid_coin is not None and quote.ask_coin is not None
                else None
            )
            relative_spread = (
                spread / quote.mid_coin
                if spread is not None and quote.mid_coin is not None and quote.mid_coin > 0.0
                else None
            )
            bid_str = f"{quote.bid_coin:.4f}" if quote.bid_coin is not None else "N/A"
            ask_str = f"{quote.ask_coin:.4f}" if quote.ask_coin is not None else "N/A"
            mid_str = f"{quote.mid_coin:.4f}" if quote.mid_coin is not None else "N/A"
            rel_str = f"{relative_spread:.4f}" if relative_spread is not None else "N/A"
            mark_str = f"{quote.deribit_mark_iv:.4f}" if quote.deribit_mark_iv is not None else "N/A"
            oi_str = f"{quote.open_interest:.0f}" if quote.open_interest is not None else "0"
            vol_str = f"{quote.volume:.0f}" if quote.volume is not None else "0"

            print(
                f"{quote.instrument_name:<24} {quote.strike:<8.0f} "
                f"{bid_str:<8} {ask_str:<8} {mid_str:<8} "
                f"{rel_str:<10} {row.lognormal_vol:<8.4f} {mark_str:<8} "
                f"{row.traditional_delta:<8.4f} {row.net_transaction_delta:<8.4f} "
                f"{oi_str:<8} {vol_str:<8}"
            )

    plot_delta_rows(
        rows,
        Path(args.output),
        snapshot_id,
        len(dropped),
        args.steps,
        model_forward,
    )

    print(
        f"snapshot={snapshot_id} expiry={selected[0].expiration_timestamp} "
        f"kept={len(rows)} dropped={len(dropped)} output={args.output}"
    )
    for instrument_name, reason in dropped:
        print(f"dropped {instrument_name}: {reason}")


if __name__ == "__main__":
    main()
