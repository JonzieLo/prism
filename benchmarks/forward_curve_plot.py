import argparse
import asyncio
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from benchmarks.chain_report import format_accounting_report, format_expiry_comparison_table
from deribit.config import SnapshotUniversalConfig
from deribit.forward_curve import *
from deribit.forwards import BasisStatus
from deribit.segmentation import build_liquidity_segmentation, format_moneyness_table, build_moneyness_segmentation
from deribit.store import SnapshotStore
from deribit.ws_client import DeribitWSClient


async def fetch_snapshot(
    store: SnapshotStore,
    currency: str,
    testnet: bool,
) -> int:
    client = DeribitWSClient(testnet=testnet)
    try:
        snapshot = await client.fetch_snapshot_data(
            SnapshotUniversalConfig(currency=currency)
        )
    finally:
        await client.close()
    return store.save_snapshot(currency, snapshot)


def plot_forward_curve(
        result: ForwardCurveResult,
        output: Path,
        *,
        snapshot_id: int | str,
        diagnostic_expiry: int | None = None,
    ):
    if not result.expiry_forwards:
        raise ValueError("No Expiry forward to be plotted")
    
    comparisons = sorted(
        result.comparisons,
        key=lambda item:item.expiration_timestamp,
    )

    dates = [
        datetime.fromtimestamp(item.expiration_timestamp / 1000.0, tz=timezone.utc)
        for item in comparisons
    ]

    implied = np.array([item.implied_forward for item in comparisons])
    future_marks = np.array(
        [
            item.future_mark
            if (
                item.status != BasisStatus.INVALID_FUTURE_BOOK.value
                and item.future_mark is not None
                and math.isfinite(item.future_mark)
                and item.future_mark > 0.0
            ) else np.nan
            for item in comparisons
        ]
    )
    with plt.rc_context(
        {
            "figure.facecolor": "#F7F6F2",
            "axes.facecolor": "#F7F6F2",
            "axes.edgecolor": "#D4D1CA",
            "axes.labelcolor": "#28251D",
            "text.color": "#28251D",
            "xtick.color": "#7A7974",
            "ytick.color": "#7A7974",
            "font.size": 10,
        }
    ):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11, 12),
            layout="constrained",
            gridspec_kw={"height_ratios": [1.0, 0.8, 1.4]},
        )
        forward_ax, basis_ax, strike_ax = axes

        forward_ax.plot(
            dates,
            implied,
            color="#20808D",
            marker="o",
            linewidth=2,
            label="Options-implied median",
        )
        forward_ax.plot(
            dates,
            future_marks,
            color="#28251D",
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label="Traded future mark",
        )
        for date, item in zip(dates, comparisons):
            if (
                item.status != BasisStatus.INVALID_FUTURE_BOOK.value
                and item.future_bid is not None
                and item.future_ask is not None
            ):
                forward_ax.vlines(
                    date,
                    item.future_bid,
                    item.future_ask,
                    color="#7A7974",
                    linewidth=4,
                    alpha=0.55,
                )
        forward_ax.set_title(
            "Options-implied forwards versus traded futures\n"
            "Median inverse put-call parity estimate by expiry",
            loc="left",
        )
        forward_ax.set_ylabel("Forward price (USD)")
        forward_ax.grid(axis="y", color="#D4D1CA", linewidth=0.7)
        forward_ax.legend(frameon=False, loc="best")

        basis_dates = []
        basis_values = []
        for date, item in zip(dates, comparisons):
            if (
                item.status != BasisStatus.INVALID_FUTURE_BOOK.value
                and item.basis_bps is not None
                and math.isfinite(item.basis_bps)
            ):
                basis_dates.append(date)
                basis_values.append(item.basis_bps)
        basis_colors = [
            "#20808D" if value >= 0.0 else "#A13544"
            for value in basis_values
        ]
        basis_ax.vlines(
            basis_dates,
            0.0,
            basis_values,
            color=basis_colors,
            linewidth=1.5,
        )
        basis_ax.scatter(
            basis_dates,
            basis_values,
            color=basis_colors,
            s=34,
            zorder=3,
        )
        basis_ax.axhline(0.0, color="#28251D", linewidth=0.9)
        basis_ax.set_title(
            "Midpoint basis by expiry\n"
            "10,000 × (options-implied forward / futures mark − 1)",
            loc="left",
        )
        basis_ax.set_ylabel("Basis (bps)")
        basis_ax.grid(axis="y", color="#D4D1CA", linewidth=0.7)

        available_expiries = {
            (item.pair.underlying_index, item.pair.expiration_timestamp)
            for item in result.evaluated_pairs
            if item.diagnostic_eligible
        }

        if diagnostic_expiry is not None:
            matching = [key for key in available_expiries if key[1] == diagnostic_expiry]
            if not matching:
                raise ValueError(f"Expiry {diagnostic_expiry} has no diagnostic-eligible pairs")
            if len(matching) > 1:
                raise ValueError(f"Expiry {diagnostic_expiry} maps to multiple underlying indexes")
            underlying_index, expiry = matching[0]
        else:
            if not available_expiries:
                raise ValueError("No expiry has diagnostic-eligible pairs")
            underlying_index, expiry = max(
                available_expiries,
                key=lambda key: sum(
                    item.diagnostic_eligible
                    and (item.pair.underlying_index, item.pair.expiration_timestamp) == key
                    for item in result.evaluated_pairs
                ),
            )
        expiry_items = sorted(
            (
                item
                for item in result.evaluated_pairs
                if (item.pair.underlying_index, item.pair.expiration_timestamp)
                == (underlying_index, expiry)
            ),
            key=lambda item: item.pair.strike,
        )

        eligible = [
            item
            for item in expiry_items
            if item.diagnostic_eligible and item.point.forward_mid is not None
        ]
        excluded = [
            item
            for item in expiry_items
            if not item.diagnostic_eligible and item.point.forward_mid is not None
        ]
        for item in expiry_items:
            point = item.point
            if (
                item.synthetic_buy_eligible
                and item.synthetic_sell_eligible
                and point.synthetic_sell_forward is not None
                and point.synthetic_buy_forward is not None
            ):
                strike_ax.vlines(
                    point.strike,
                    point.synthetic_sell_forward,
                    point.synthetic_buy_forward,
                    color="#BAB9B4",
                    linewidth=1,
                    alpha=0.7,
                )
        strike_ax.scatter(
            [item.point.strike for item in eligible],
            [item.point.forward_mid for item in eligible],
            color="#20808D",
            marker="o",
            label="Diagnostic eligible",
            zorder=3,
        )
        if excluded:
            strike_ax.scatter(
                [item.point.strike for item in excluded],
                [item.point.forward_mid for item in excluded],
                color="#A13544",
                marker="x",
                label="Excluded midpoint",
                zorder=3,
            )
        comparison = next(
            item
            for item in comparisons
            if (
                item.underlying_index,
                item.expiration_timestamp,
            )
            == (underlying_index, expiry)
        )
        strike_ax.axhline(
            comparison.implied_forward,
            color="#20808D",
            linewidth=1.5,
            label="Expiry median",
        )
        if (
            comparison.status != BasisStatus.INVALID_FUTURE_BOOK.value
            and comparison.future_mark is not None
            and math.isfinite(comparison.future_mark)
            and comparison.future_mark > 0.0
        ):
            strike_ax.axhline(
                comparison.future_mark,
                color="#28251D",
                linestyle="--",
                linewidth=1.2,
                label="Future mark",
            )
        if (
            comparison.status != BasisStatus.INVALID_FUTURE_BOOK.value
            and comparison.future_bid is not None
            and comparison.future_ask is not None
        ):
            strike_ax.axhspan(
                comparison.future_bid,
                comparison.future_ask,
                color="#7A7974",
                alpha=0.12,
                label="Future bid-ask",
            )
        strike_ax.set_title(
            f"Strike-level parity diagnostics: {comparison.underlying_index}\n"
            "Vertical ranges are strike-specific synthetic sell-to-buy prices",
            loc="left",
        )
        strike_ax.set_xlabel("Strike (USD)")
        strike_ax.set_ylabel("Implied forward (USD)")
        strike_ax.grid(color="#D4D1CA", linewidth=0.7)
        strike_ax.legend(frameon=False, loc="best", ncols=2)

        for axis in (forward_ax, basis_ax):
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
            axis.tick_params(axis="x", rotation=25)

        figure.suptitle(
            f"PRISM forward-curve diagnostics | snapshot {snapshot_id}",
            fontsize=16,
            fontweight="bold",
            x=0.01,
            ha="left",
        )
        figure.text(
            0.01,
            -0.01,
            (
                "Midpoints are diagnostic estimates, not executable prices. "
                "Price crosses do not include fees, slippage, margin, or "
                "contract-size matching."
            ),
            fontsize=8.5,
            color="#7A7974",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)


def print_snapshot_provenance(snapshot: dict) -> None:
    opt_ns = snapshot.get("options", {}).get("received_at_ns")
    fut_ns = snapshot.get("futures", {}).get("received_at_ns")
    idx_ns = snapshot.get("index", {}).get("received_at_ns")

    print("\n=== SNAPSHOT PROVENANCE & TIMING ===")
    if opt_ns and fut_ns and idx_ns:
        diff_ms = (fut_ns - opt_ns) / 1_000_000.0
        window_ms = (max(opt_ns, fut_ns, idx_ns) - min(opt_ns, fut_ns, idx_ns)) / 1_000_000.0
        print(f"Options Summary Received: {opt_ns}")
        print(f"Futures Summary Received: {fut_ns}")
        print(f"Options-to-Futures Delta: {diff_ms:+.2f} ms")
        print(f"Total Snapshot Capture Window: {window_ms:.2f} ms")
    else:
        print("Timestamp provenance metadata incomplete in snapshot payload.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and plot an inverse-option forward curve."
    )
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--snapshot-id", type=int)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--expiry", type=int)
    parser.add_argument("--output", default="figs/forward_curve.png")
    return parser.parse_args()


if __name__ == "__main__":
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
            raise SystemExit("No stored snapshot; use --fetch or --snapshot-id")
        snapshot = store.load_snapshot(snapshot_id)

    if snapshot is None:
        raise SystemExit(f"Snapshot {snapshot_id} does not exist")

    result = build_forward_curve(snapshot)
    print_snapshot_provenance(snapshot)
    print("\n" + format_accounting_report(result))
    print("\n" + format_expiry_comparison_table(result))

    print("\n=== MONEYNESS SEGMENTATION (ALL EXPIRIES) ===")
    for expiry_forward in result.expiry_forwards:
        metrics = build_moneyness_segmentation(
            result.evaluated_pairs,
            expiry_forward,
        )
        print(f"\nExpiry: {expiry_forward.underlying_index}")
        print(format_moneyness_table(metrics))

    liq_metrics = build_liquidity_segmentation(result.evaluated_pairs, result.expiry_forwards)
    if liq_metrics:
        print("\n=== LIQUIDITY GROUP SEGMENTATION ===")
        print(
            f"{'Group':<22} {'Pairs':<6} {'Elig%':<8} {'MedSprd':<9} "
            f"{'MedResUSD':<10} {'MedSynthWidth':<12}"
        )
        print("-" * 70)
        for lm in liq_metrics:
            sprd_str = f"{lm.median_relative_spread:.2%}" if lm.median_relative_spread else "N/A"
            res_str = f"{lm.median_abs_parity_residual:.2f}" if lm.median_abs_parity_residual else "N/A"
            width_str = f"{lm.median_synthetic_interval_width:.2f}" if lm.median_synthetic_interval_width else "N/A"
            print(
                f"{lm.group_name:<22} {lm.pair_count:<6} "
                f"{lm.midpoint_eligible_fraction:<8.2%} {sprd_str:<9} "
                f"{res_str:<10} {width_str:<12}"
            )

    plot_forward_curve(
        result,
        Path(args.output),
        snapshot_id=snapshot_id,
        diagnostic_expiry=args.expiry,
    )
    print(f"\nFigure saved to: {args.output}")