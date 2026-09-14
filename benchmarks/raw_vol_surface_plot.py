import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deribit.config import SnapshotUniversalConfig
from deribit.forward_curve import build_forward_curve
from deribit.store import SnapshotStore
from deribit.surface.raw_iv import SURFACE_MODEL_NAMES,RawIVResult,build_raw_iv_points
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


def plot_raw_iv(result: RawIVResult, output: Path) -> None:
    if not result.points:
        raise ValueError("No implied-volatility points were recovered")

    grouped = defaultdict(list)
    for point in result.points:
        grouped[point.expiration_timestamp].append(point)

    expiries = sorted(grouped)
    colors = plt.colormaps["viridis"](
        np.linspace(0.08, 0.92, len(expiries))
    )
    is_normal = result.model_name == "bachelier"

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
        figure, axis = plt.subplots(
            figsize=(11, 7),
            layout="constrained",
        )

        for color, expiry in zip(colors, expiries):
            points = sorted(grouped[expiry], key=lambda item: item.strike)
            strikes = np.array([item.strike for item in points])
            vols = np.array([item.implied_vol for item in points])
            if not is_normal:
                vols *= 100.0
            label = datetime.fromtimestamp(
                expiry / 1000.0,
                tz=timezone.utc,
            ).strftime("%d %b %Y")
            axis.plot(
                strikes,
                vols,
                color=color,
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                label=label,
            )

        model_label = result.model_name.replace("_", " ").title()
        if result.model_name == "inverse":
            model_label = "Inverse premium via Black-76"
        axis.set_title(
            f"Raw {model_label} implied volatility by expiry\n"
            "Canonical OTM midpoints; lines only connect observed strikes",
            loc="left",
        )
        axis.set_xlabel("Strike K (USD)")
        axis.set_ylabel(
            "Normal IV (USD / sqrt(year))"
            if is_normal
            else "Lognormal implied volatility (%)"
        )
        axis.grid(axis="y", color="#D4D1CA", linewidth=0.7)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            title="Expiry",
            frameon=False,
            ncols=2 if len(expiries) > 6 else 1,
            fontsize=8,
        )

        figure.text(
            0.01,
            -0.01,
            (
                f"Snapshot {result.snapshot_id} | "
                f"model={result.model_name} | "
                f"points={len(result.points)} | "
                f"dropped={len(result.drops)}"
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
        description=(
            "Plot raw model-implied volatility smiles from one snapshot."
        )
    )
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--snapshot-id", type=int)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument(
        "--model",
        choices=SURFACE_MODEL_NAMES,
        default="black76",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--output",
        default="figs/raw_vol_smiles.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    store = SnapshotStore(args.db)

    if args.fetch:
        snapshot_id = asyncio.run(
            fetch_snapshot(store, args.currency, args.testnet)
        )
    elif args.snapshot_id is not None:
        snapshot_id = args.snapshot_id
    else:
        snapshot_id = store.latest_snapshot_id(args.currency)
        if snapshot_id is None:
            raise SystemExit(
                "No stored snapshot; use --fetch or --snapshot-id"
            )

    snapshot = store.load_snapshot(snapshot_id)
    if snapshot is None:
        raise SystemExit(f"Snapshot {snapshot_id} does not exist")

    curve = build_forward_curve(snapshot)
    result = build_raw_iv_points(
        snapshot_id,
        list(curve.quotes),
        list(curve.expiry_forwards),
        args.model,
        binomial_steps=args.steps,
    )
    plot_raw_iv(result, Path(args.output))

    print(
        f"snapshot={snapshot_id} model={args.model} "
        f"points={len(result.points)} drops={len(result.drops)} "
        f"output={args.output}"
    )
    for drop in result.drops:
        print(
            f"dropped expiry={drop.expiration_timestamp} "
            f"strike={drop.strike:g}: {drop.reason}"
        )