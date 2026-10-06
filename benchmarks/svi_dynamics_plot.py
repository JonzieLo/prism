import argparse
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deribit.forward_curve import build_forward_curve
from deribit.store import SnapshotStore
from deribit.surface.filter_policy import SurfaceFilterPolicy
from deribit.surface.pipeline import build_surface_observations
from deribit.surface.results import CalibratedSmile
from deribit.surface.term_structure import (
    DEFAULT_MINIMUM_OBSERVATIONS,
    TimeDerivativeSegment,
    build_time_derivative_segments,
    fit_snapshot_term_structure,
    spot_backbone,
)

_VIOLATION_COLOR = "#C1121F"
_NEUTRAL = "#7A7974"

_RC = {
    "figure.facecolor": "#F7F6F2",
    "axes.facecolor": "#F7F6F2",
    "axes.edgecolor": "#D4D1CA",
    "axes.labelcolor": "#28251D",
    "text.color": "#28251D",
    "xtick.color": _NEUTRAL,
    "ytick.color": _NEUTRAL,
    "font.size": 10,
}

_STICKINESS_FAMILY = (0.0, 1.0, 1.5)


def _get_viridis_colors(count: int) -> list:
    """Matches the exact viridis gradient from raw_vol_surface_plot.py (dark purple -> yellow)."""
    if count == 1:
        return [plt.colormaps["viridis"](0.10)]
    return list(plt.colormaps["viridis"](np.linspace(0.08, 0.92, count)))


def _expiry_label(smile: CalibratedSmile) -> str:
    return datetime.fromtimestamp(
        smile.expiration_timestamp / 1000.0, tz=timezone.utc
    ).strftime("%d %b %Y")


def _expiry_label_ts(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1000.0, tz=timezone.utc
    ).strftime("%d %b %Y")


def _style_axes(axes) -> None:
    for axis in axes:
        axis.axhline(0.0, color=_NEUTRAL, linewidth=0.9)
        axis.axvline(0.0, color=_NEUTRAL, linewidth=0.9, linestyle=":")
        axis.grid(axis="y", color="#D4D1CA", linewidth=0.7)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=7.5)


def _save(figure, output: Path, footer: str) -> None:
    figure.text(0.01, -0.01, footer, fontsize=7.6, color=_NEUTRAL)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


# ======================================================================
# Figure 1: Spot Dynamics
# ======================================================================


def plot_spot_dynamics(
    smiles: list[CalibratedSmile],
    output: Path,
    *,
    snapshot_id: int,
    skew_stickiness_ratio: float,
    forward_spot_elasticity: float,
    n_points: int = 201,
) -> None:
    if not smiles:
        raise ValueError("Need at least one calibrated smile")

    colors = _get_viridis_colors(len(smiles))

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, layout="constrained")
        skew_axis, response_axis, pnl_axis = axes

        for color, smile in zip(colors, smiles):
            backbone = spot_backbone(
                smile,
                skew_stickiness_ratio=skew_stickiness_ratio,
                forward_spot_elasticity=forward_spot_elasticity,
                n_points=n_points,
            )
            arrays = backbone.as_arrays()
            k = arrays["log_moneyness"]

            atm_skew = float(smile.parameters.dvol_dk(0.0, smile.tau) * 100.0)
            atm_skew_normalised = atm_skew * np.sqrt(smile.tau)
            label = (
                f"{_expiry_label(smile)} (tau={smile.tau:.3f}y, "
                f"ATM skew={atm_skew:.1f}, sqrt(tau)-adj={atm_skew_normalised:.2f})"
            )

            skew_axis.plot(k, arrays["dvol_dk"] * 100.0, color=color, linewidth=1.8, label=label)
            response_axis.plot(
                k, arrays["vol_points_per_1pct_spot"], color=color, linewidth=1.8,
                label=f"{_expiry_label(smile)} (R={skew_stickiness_ratio:g})"
            )
            pnl_axis.plot(
                k, arrays["vol_pnl_per_1pct_spot_bp_of_forward"], color=color, linewidth=1.8,
                label=_expiry_label(smile)
            )

        front = smiles[0]
        for ratio, dash in zip(_STICKINESS_FAMILY, ((4, 2), (1, 2), (6, 2, 1, 2))):
            if ratio == skew_stickiness_ratio:
                continue
            backbone = spot_backbone(
                front,
                skew_stickiness_ratio=ratio,
                forward_spot_elasticity=forward_spot_elasticity,
                n_points=n_points,
            )
            arrays = backbone.as_arrays()
            response_axis.plot(
                arrays["log_moneyness"],
                arrays["vol_points_per_1pct_spot"],
                color=_NEUTRAL,
                linewidth=1.2,
                dashes=dash,
                alpha=0.8,
                label=f"{_expiry_label(front)} (R={ratio:g})",
            )

        skew_axis.set_title(f"Snapshot {snapshot_id}: SVI spot dynamics", loc="left")
        skew_axis.set_ylabel("d(sigma)/dk\n(vol points per unit k)")
        response_axis.set_ylabel("Fixed-strike IV response to +1% spot\n(vol points)")
        pnl_axis.set_ylabel("Vol P&L from +1% spot: vega x d(sigma)\n(bp of forward notional)")
        pnl_axis.set_xlabel("Log-moneyness k = ln(K / F)")

        _style_axes(axes)
        footer = (
            f"Panels 1-2 are proportional: d(sigma_K)/d(lnS) = eps (R - 1) d(sigma)/dk.\n"
            f"Panel 3 incorporates Vega weighting. Plotted at R={skew_stickiness_ratio:g}, eps={forward_spot_elasticity:g}."
        )
        _save(figure, output, footer)


# ======================================================================
# Figure 2: Time Dynamics
# ======================================================================


def plot_time_dynamics(
    segments: list[TimeDerivativeSegment],
    output: Path,
    *,
    snapshot_id: int,
) -> None:
    if not segments:
        raise ValueError("Need at least one adjacent expiry pair")

    colors = _get_viridis_colors(len(segments))

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, layout="constrained")
        decay_axis, decomposition_axis, calendar_axis = axes

        for color, segment in zip(colors, segments):
            arrays = segment.as_arrays()
            k = arrays["log_moneyness"]
            pair_label = (
                f"{_expiry_label_ts(segment.expiry_near)} -> "
                f"{_expiry_label_ts(segment.expiry_far)} "
                f"(at tau={segment.tau_evaluated:.3f}y)"
            )

            decay_axis.plot(k, arrays["vol_points_per_calendar_day"], color=color, linewidth=1.8, label=pair_label)

            scale = -100.0 / 365.0
            decomposition_axis.plot(k, arrays["accumulation_term"] * scale, color=color, linewidth=1.6, label=f"{pair_label}: accumulation")
            decomposition_axis.plot(k, arrays["annualisation_term"] * scale, color=color, linewidth=1.6, linestyle="--", alpha=0.8, label=f"{pair_label}: annualisation")

            margin = arrays["calendar_margin_relative"] * 100.0
            calendar_axis.plot(k, margin, color=color, linewidth=1.8, label=f"{pair_label}: min={margin.min():.3f}%")
            violating = margin < 0.0
            if np.any(violating):
                calendar_axis.scatter(k[violating], margin[violating], color=_VIOLATION_COLOR, marker="x", s=40, zorder=3)

        decay_axis.set_title(f"Snapshot {snapshot_id}: SVI time dynamics", loc="left")
        decay_axis.set_ylabel("IV drift per calendar day held\n(vol points)")
        decomposition_axis.set_ylabel("Decomposition of IV drift\n(vol points per day)")
        calendar_axis.set_ylabel("Calendar no-arb margin\n(w_far - w_near) / w_near (%)")
        calendar_axis.set_xlabel("Log-moneyness k = ln(K / F)")

        _style_axes(axes)
        footer = "d(sigma)/d(tau) = dw_dtau / (2 sigma tau) - sigma / (2 tau) from linear-in-tau total variance interpolation."
        _save(figure, output, footer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SVI spot and time dynamics for one snapshot.")
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--snapshot-id", type=int)
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--number-of-starts", type=int, default=16)
    parser.add_argument("--max-abs-k", type=float, default=0.75)
    parser.add_argument("--max-relative-spread", type=float)
    parser.add_argument("--min-observations", type=int, default=DEFAULT_MINIMUM_OBSERVATIONS)
    parser.add_argument("--skew-stickiness-ratio", type=float, default=0.0)
    parser.add_argument("--forward-spot-elasticity", type=float, default=1.0)
    parser.add_argument("--evaluate-at", choices=("near", "far", "midpoint"), default="near")
    parser.add_argument("--spot-output", default="figs/svi_spot_dynamics.png")
    parser.add_argument("--time-output", default="figs/svi_time_dynamics.png")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    store = SnapshotStore(args.db)
    snapshot_id = args.snapshot_id if args.snapshot_id is not None else store.latest_snapshot_id(args.currency)
    if snapshot_id is None:
        raise SystemExit("No stored snapshot")
    snapshot = store.load_snapshot(snapshot_id)
    if snapshot is None:
        raise SystemExit(f"Snapshot {snapshot_id} does not exist")

    curve = build_forward_curve(snapshot)
    surface = build_surface_observations(
        snapshot_id, list(curve.quotes), list(curve.expiry_forwards), SurfaceFilterPolicy(max_relative_spread=args.max_relative_spread)
    )

    observations_by_expiry: dict[int, list] = defaultdict(list)
    for item in surface.observations:
        if abs(item.log_moneyness) <= args.max_abs_k:
            observations_by_expiry[item.expiration_timestamp].append(item)

    smiles = fit_snapshot_term_structure(
        observations_by_expiry, number_of_starts=args.number_of_starts, min_observations=args.min_observations
    )
    if not smiles:
        raise SystemExit("No expiry survived calibration")

    plot_spot_dynamics(
        smiles, Path(args.spot_output), snapshot_id=snapshot_id, skew_stickiness_ratio=args.skew_stickiness_ratio, forward_spot_elasticity=args.forward_spot_elasticity
    )

    segments = build_time_derivative_segments(smiles, evaluate_at=args.evaluate_at)
    if segments:
        plot_time_dynamics(segments, Path(args.time_output), snapshot_id=snapshot_id)
    else:
        print("No adjacent pair shared a log-moneyness range; skipping time-dynamics figure.")

    print(f"snapshot={snapshot_id} fitted {len(smiles)} expiries")
    print(f"figures saved to: {args.spot_output}, {args.time_output}")