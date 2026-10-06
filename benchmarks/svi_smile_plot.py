import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deribit.forward_curve import build_forward_curve
from deribit.store import SnapshotStore
from deribit.surface.arbitrage import assert_butterfly_free
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.filter_policy import SurfaceFilterPolicy
from deribit.surface.pipeline import build_surface_observations
from deribit.surface.results import CalibratedSmile


def _select_expiry(
    observations,
    expiration_timestamp: int | None,
):
    grouped = defaultdict(list)
    for item in observations:
        grouped[item.expiration_timestamp].append(item)
    if expiration_timestamp is not None:
        selected = grouped.get(expiration_timestamp, [])
        if not selected:
            raise ValueError(
                f"No surface observations for expiry {expiration_timestamp}"
            )
        return selected
    if not grouped:
        raise ValueError("No eligible surface observations")
    selected_expiry = max(grouped, key=lambda key: len(grouped[key]))
    print(
        f"No --expiry supplied; selected {selected_expiry} because it has "
        f"the most eligible observations ({len(grouped[selected_expiry])})."
    )
    return grouped[selected_expiry]


def plot_svi_smile(
    smile: CalibratedSmile,
    output: Path,
    *,
    snapshot_timestamp_ns: int | None = None,
    plot_space: str = 'implied_vol',
) -> None:
    if plot_space not in {'implied_vol', 'total_variance'}:
        raise ValueError(f"Unsupported plot space: {plot_space}")
    residuals = sorted(
        smile.residuals,
        key=lambda item: item.log_moneyness,
    )
    observed_k = np.array([item.log_moneyness for item in residuals])
    if plot_space == "implied_vol":
        market_w = np.array(
            [item.market_iv * 100.0 for item in residuals]
        )
        fitted_residuals = np.array(
            [item.iv_residual * 100.0 for item in residuals]
        )
    else:
        market_w = np.array(
            [item.market_total_variance for item in residuals]
        )
        fitted_residuals = np.array(
            [item.total_variance_residual for item in residuals]
        )
    option_types = np.array([item.option_type for item in residuals])
    inside_spread = np.array(
        [
            item.spread_normalized_residual is not None
            and abs(item.spread_normalized_residual) <= 1.0
            for item in residuals
        ]
    )

    padding = max(0.10, 0.20 * (smile.observed_k_max - smile.observed_k_min))
    k_grid = np.linspace(
        smile.observed_k_min - padding,
        smile.observed_k_max + padding,
        501,
    )
    if plot_space == "implied_vol":
        fitted_w = (
            smile.parameters.implied_vol(k_grid, smile.tau) * 100.0
        )
    else:
        fitted_w = smile.parameters.total_variance(k_grid)
    in_observed_range = (
        (k_grid >= smile.observed_k_min)
        & (k_grid <= smile.observed_k_max)
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
        figure, (smile_axis, residual_axis) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )

        smile_axis.plot(
            k_grid[in_observed_range],
            fitted_w[in_observed_range],
            color="#20808D",
            linewidth=2.2,
            label="SVI fit",
        )
        left_wing = k_grid < smile.observed_k_min
        right_wing = k_grid > smile.observed_k_max
        smile_axis.plot(
            k_grid[left_wing],
            fitted_w[left_wing],
            color="#20808D",
            linewidth=1.4,
            linestyle="--",
            alpha=0.7,
            label="SVI extrapolation",
        )
        smile_axis.plot(
            k_grid[right_wing],
            fitted_w[right_wing],
            color="#20808D",
            linewidth=1.4,
            linestyle="--",
            alpha=0.7,
        )
        for option_type, marker, label in (
            ("put", "o", "OTM puts"),
            ("call", "^", "OTM calls"),
        ):
            mask = option_types == option_type
            smile_axis.scatter(
                observed_k[mask],
                market_w[mask],
                color="#28251D",
                marker=marker,
                s=28,
                zorder=3,
                label=label,
            )

        residual_axis.scatter(
            observed_k[inside_spread],
            fitted_residuals[inside_spread],
            color="#20808D",
            marker="o",
            s=28,
            label="Fitted price inside bid-ask",
        )
        residual_axis.scatter(
            observed_k[~inside_spread],
            fitted_residuals[~inside_spread],
            color="#A13544",
            marker="x",
            s=38,
            label="Fitted price outside bid-ask",
        )

        expiry = datetime.fromtimestamp(
            smile.expiration_timestamp / 1000.0,
            tz=timezone.utc,
        ).strftime("%d %b %Y")
        report = smile.arbitrage_report
        smile_axis.set_title(
            f"BTC implied-volatility smile: {expiry}\n" if plot_space == "implied_vol" else f"BTC total-variance smile: {expiry}\n"
            "Market midpoints versus butterfly-checked raw SVI",
            loc="left",
        )
        smile_axis.set_ylabel(
            "Implied volatility (%)"
            if plot_space == "implied_vol"
            else "Total variance w"
        )
        residual_axis.set_ylabel(
            "Market − SVI (vol points)"
            if plot_space == "implied_vol"
            else "Market − SVI w"
        )
        residual_axis.set_xlabel("Log-moneyness k = ln(K / F)")
        residual_axis.axhline(0.0, color="#7A7974", linewidth=0.9)

        for axis in (smile_axis, residual_axis):
            axis.axvline(
                0.0,
                color="#7A7974",
                linewidth=0.9,
                linestyle=":",
            )
            axis.grid(axis="y", color="#D4D1CA", linewidth=0.7)
            axis.grid(axis="x", visible=False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.legend(frameon=False, fontsize=8)

        timestamp = (
            datetime.fromtimestamp(
                snapshot_timestamp_ns / 1_000_000_000.0,
                tz=timezone.utc,
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
            if snapshot_timestamp_ns is not None
            else "timestamp unavailable"
        )
        p = smile.parameters
        footer = (
            f"Snapshot {smile.snapshot_id} ({timestamp}) | "
            f"F={smile.forward:,.2f} | n={smile.observation_count} | "
            f"RMSE(w)={smile.rmse_total_variance:.3e} | "
            f"a={p.a:.5f}, b={p.b:.5f}, rho={p.rho:.4f}, "
            f"m={p.m:.4f}, eta={p.eta:.4f} | "
            f"min g(k)={report.minimum_density_condition:.3e} | "
            f"butterfly={'PASS' if report.butterfly_free else 'FAIL'}"
        )
        figure.text(
            0.01,
            -0.01,
            footer,
            fontsize=7.8,
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
        description="Fit and plot one butterfly-checked SVI smile."
    )
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--snapshot-id", type=int)
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--expiry", type=int)
    parser.add_argument("--number-of-starts", type=int, default=16)
    parser.add_argument("--max-abs-k", type=float, default=0.75)
    parser.add_argument("--max-relative-spread", type=float)
    parser.add_argument(
        "--plot-space",
        choices=("implied_vol", "total_variance"),
        default="implied_vol",
    )
    parser.add_argument(
        "--output",
        default="figs/svi_smile.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    store = SnapshotStore(args.db)
    snapshot_id = (
        args.snapshot_id
        if args.snapshot_id is not None
        else store.latest_snapshot_id(args.currency)
    )
    if snapshot_id is None:
        raise SystemExit("No stored snapshot")
    snapshot = store.load_snapshot(snapshot_id)
    if snapshot is None:
        raise SystemExit(f"Snapshot {snapshot_id} does not exist")

    curve = build_forward_curve(snapshot)
    surface = build_surface_observations(
        snapshot_id,
        list(curve.quotes),
        list(curve.expiry_forwards),
        SurfaceFilterPolicy(
            max_relative_spread=args.max_relative_spread,
        ),
    )
    selected = _select_expiry(
        surface.observations,
        args.expiry,
    )
    selected = [
        item
        for item in selected
        if abs(item.log_moneyness) <= args.max_abs_k
    ]
    smile = calibrate_svi_slice(
        selected,
        number_of_starts=args.number_of_starts,
    )
    assert_butterfly_free(smile.arbitrage_report)

    metadata = store.load_snapshot_metadata(snapshot_id)
    plot_svi_smile(
        smile,
        Path(args.output),
        snapshot_timestamp_ns=(
            metadata.timestamp_ns if metadata is not None else None
        ),
        plot_space=args.plot_space,
    )

    p = smile.parameters
    report = smile.arbitrage_report
    reason_counts = Counter(item.reason.value for item in surface.exclusions)
    print(
        f"snapshot={snapshot_id} expiry={smile.expiration_timestamp} "
        f"points={smile.observation_count} "
        f"rmse_w={smile.rmse_total_variance:.6e} "
        f"butterfly={report.butterfly_free}"
    )
    print(
        f"SVI a={p.a:.10f} b={p.b:.10f} rho={p.rho:.10f} "
        f"m={p.m:.10f} eta={p.eta:.10f}"
    )
    print(
        f"min_w={report.minimum_total_variance:.6e} "
        f"min_g={report.minimum_density_condition:.6e} "
        f"min_convexity={report.minimum_convexity_margin:.6e}"
    )
    if reason_counts:
        print("surface exclusions:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")
    print(f"figure saved to: {args.output}")
