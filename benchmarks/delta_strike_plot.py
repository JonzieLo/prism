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

        try:
            ##temp: setting foward & rate to Deribit forward & rate
            forward = quote.api_forward
            rate = quote.api_rate
            lognormal_vol = black76.implied_vol(
                mid_usd,
                # quote.forward,
                forward,
                quote.strike,
                quote.tau,
                # quote.rate,
                rate,
                quote.option_type,
            )
            normal_vol = bachelier.implied_vol(
                mid_usd,
                # quote.forward,
                forward,
                quote.strike,
                quote.tau,
                # quote.rate,
                rate,
                quote.option_type,
            )
            black76_greeks = black76.greeks(
                # quote.forward,
                forward,
                quote.strike,
                quote.tau,
                lognormal_vol,
                # quote.rate,
                rate,
                quote.option_type,
            )
            binomial_greeks = binomial.greeks(
                # quote.forward,
                forward,
                quote.strike,
                quote.tau,
                lognormal_vol,
                # quote.rate,
                rate,
                quote.option_type,
            )
            bachelier_greeks = bachelier.greeks(
                # quote.forward,
                forward,
                quote.strike,
                quote.tau,
                normal_vol,
                # quote.rate,
                rate,
                quote.option_type,
            )
            cash_price = black76.price(
                # quote.forward,
                forward,
                quote.strike,
                quote.tau,
                lognormal_vol,
                # quote.rate,
                rate,
                quote.option_type,
            )
            inverse = from_forward_greeks(
                cash_price,
                black76_greeks,
                quote.index_price,
                # quote.forward,
                forward,
            )
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
            )
        )

    return rows, dropped


def plot_delta_rows(
    rows: list[DeltaRow],
    output: Path,
    snapshot_id: int | str,
    dropped_count: int,
    steps: int,
) -> None:
    if not rows:
        raise ValueError("no valid option rows remain after IV inversion")

    strikes = np.array([row.quote.strike for row in rows])
    traditional = np.array([row.traditional_delta for row in rows])
    black76 = np.array([row.black76_delta for row in rows])
    binomial = np.array([row.binomial_delta for row in rows])
    bachelier = np.array([row.bachelier_delta for row in rows])
    ntd = np.array([row.net_transaction_delta for row in rows])

    mark_differences = [
        abs(row.lognormal_vol - row.quote.deribit_mark_iv)
        for row in rows
        if row.quote.deribit_mark_iv is not None
    ]
    median_mark_difference = (
        float(np.median(mark_differences)) if mark_differences else math.nan
    )
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
            strikes,
            traditional,
            color="#20808D",
            marker="o",
            markersize=3,
            lw=2.2,
            label="Spot-equivalent traditional delta",
        )
        top.plot(
            strikes,
            black76,
            color="#A84B2F",
            marker="s",
            markersize=3,
            lw=1.8,
            ls="--",
            label="Black–76 forward delta",
        )
        top.plot(
            strikes,
            binomial,
            color="#1B474D",
            lw=1.4,
            label=f"CRR forward delta ({steps} steps)",
        )
        top.plot(
            strikes,
            bachelier,
            color="#7A39BB",
            lw=1.8,
            ls="-.",
            label="Price-matched Bachelier delta",
        )
        top.set_ylabel("Delta")
        top.set_title(
            "Traditional deltas use market-mid implied volatility\n"
            "Deribit mark IV is retained only as a cross-check",
            loc="left",
        )
        top.legend(frameon=False, ncol=2, loc="best")

        bottom.plot(
            strikes,
            ntd,
            color="#DA7101",
            marker="o",
            markersize=3,
            lw=2.4,
            label="Inverse Net Transaction Delta",
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
            # Mark Spot/Index Price
            axis.axvline(
                first.index_price, 
                color="#7A7974", 
                ls=":", 
                lw=1.2, 
                label=f"Index X (${first.index_price:,.0f})"
            )
            # Mark Forward Price
            axis.axvline(
                first.api_forward, 
                color="#28251D", 
                ls="--", 
                lw=1.2, 
                label=f"Forward F (${first.api_forward:,.0f})"
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
        cross_check = (
            f"{median_mark_difference:.4f}"
            if math.isfinite(median_mark_difference)
            else "n/a"
        )
        figure.text(
            0.01,
            -0.02,
            (
                f"Expiry {expiry} | X={first.index_price:,.2f} | "
                f"F={first.api_forward:,.2f} | kept={len(rows)} | "
                f"dropped={dropped_count} | median |own IV − mark IV|="
                f"{cross_check}"
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

    chain = option_chain_from_snapshot(snapshot)
    selected = select_expiry(chain, args.option_type, args.expiry)
    rows, dropped = build_delta_rows(selected, args.steps)
    plot_delta_rows(
        rows,
        Path(args.output),
        snapshot_id,
        len(dropped),
        args.steps,
    )

    print(
        f"snapshot={snapshot_id} expiry={selected[0].expiration_timestamp} "
        f"kept={len(rows)} dropped={len(dropped)} output={args.output}"
    )
    for instrument_name, reason in dropped:
        print(f"dropped {instrument_name}: {reason}")


if __name__ == "__main__":
    main()
