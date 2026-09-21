import argparse
import logging
from collections import defaultdict
from datetime import datetime, timezone
from itertools import cycle, islice
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

_COLORS = [
    "#20808D",
    "#A13544",
    "#B98A2E",
    "#5B5F97",
    "#4E7C4E",
    "#8D5B9A",
]
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

# R = 0 sticky delta, R = 1 sticky strike, R = 1.5 typical index regime.
_STICKINESS_FAMILY = (0.0, 1.0, 1.5)


def _colors_for(count: int) -> list[str]:
    """Cycle rather than zip-truncate.

    zip(_COLORS, smiles) silently dropped every expiry past the sixth.
    """
    return list(islice(cycle(_COLORS), count))


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
    figure.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


# ======================================================================
# Figure 1: spot dynamics
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

    colors = _colors_for(len(smiles))

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                                    layout="constrained")
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

            # Report the tau-normalised ATM skew in the legend so the
            # 1/sqrt(tau) scaling of the raw curves is visible rather than
            # silently driving the cross-expiry comparison.
            atm_skew = float(
                smile.parameters.dvol_dk(0.0, smile.tau) * 100.0
            )
            atm_skew_normalised = atm_skew * np.sqrt(smile.tau)
            label = (
                f"{_expiry_label(smile)} (tau={smile.tau:.3f}y, "
                f"ATM skew={atm_skew:.1f}, "
                f"sqrt(tau)-adj={atm_skew_normalised:.2f})"
            )

            skew_axis.plot(
                k,
                arrays["dvol_dk"] * 100.0,
                color=color,
                linewidth=1.8,
                label=label,
            )
            response_axis.plot(
                k,
                arrays["vol_points_per_1pct_spot"],
                color=color,
                linewidth=1.8,
                label=f"{_expiry_label(smile)} (R={skew_stickiness_ratio:g})",
            )
            pnl_axis.plot(
                k,
                arrays["vol_pnl_per_1pct_spot_bp_of_forward"],
                color=color,
                linewidth=1.8,
                label=_expiry_label(smile),
            )

        # Stickiness family on the front expiry only, to show how much of
        # the answer is assumption rather than data.
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

        skew_axis.set_title(
            f"Snapshot {snapshot_id}: SVI spot dynamics", loc="left"
        )
        skew_axis.set_ylabel(
            "d(sigma)/dk\n(vol points per unit k)"
        )
        response_axis.set_ylabel(
            "Fixed-strike IV response to +1% spot\n(vol points)"
        )
        pnl_axis.set_ylabel(
            "Vol P&L from +1% spot: vega x d(sigma)\n(bp of forward notional)"
        )
        pnl_axis.set_xlabel("Log-moneyness k = ln(K / F)")

        _style_axes(axes)

        footer = (
            f"Panels 1-2 are proportional by construction: any translation-family "
            f"stickiness rule gives d(sigma_K)/d(lnS) = eps (R - 1) d(sigma)/dk. "
            f"Panel 3 is the independent one (vega weighting).\n"
            f"R = skew-stickiness ratio: 0 sticky-delta, 1 sticky-strike, "
            f"~1.5 typical index. Plotted at R={skew_stickiness_ratio:g}, "
            f"eps = d lnF / d lnS = {forward_spot_elasticity:g} (assumed, not measured)."
        )
        _save(figure, output, footer)


# ======================================================================
# Figure 2: time dynamics
# ======================================================================


def plot_time_dynamics(
    segments: list[TimeDerivativeSegment],
    output: Path,
    *,
    snapshot_id: int,
) -> None:
    if not segments:
        raise ValueError(
            "Need at least one adjacent expiry pair with a shared "
            "log-moneyness range"
        )

    colors = _colors_for(len(segments))

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                                    layout="constrained")
        decay_axis, decomposition_axis, calendar_axis = axes

        for color, segment in zip(colors, segments):
            arrays = segment.as_arrays()
            k = arrays["log_moneyness"]
            pair_label = (
                f"{_expiry_label_ts(segment.expiry_near)} -> "
                f"{_expiry_label_ts(segment.expiry_far)} "
                f"(at tau={segment.tau_evaluated:.3f}y)"
            )

            decay_axis.plot(
                k,
                arrays["vol_points_per_calendar_day"],
                color=color,
                linewidth=1.8,
                label=pair_label,
            )

            scale = -100.0 / 365.0
            decomposition_axis.plot(
                k,
                arrays["accumulation_term"] * scale,
                color=color,
                linewidth=1.6,
                label=f"{pair_label}: variance accumulation",
            )
            decomposition_axis.plot(
                k,
                arrays["annualisation_term"] * scale,
                color=color,
                linewidth=1.6,
                linestyle="--",
                alpha=0.8,
                label=f"{pair_label}: annualisation (1/tau)",
            )

            margin = arrays["calendar_margin_relative"] * 100.0
            calendar_axis.plot(
                k,
                margin,
                color=color,
                linewidth=1.8,
                label=f"{pair_label}: min={margin.min():.3f}%",
            )
            violating = margin < 0.0
            if np.any(violating):
                calendar_axis.scatter(
                    k[violating],
                    margin[violating],
                    color=_VIOLATION_COLOR,
                    marker="x",
                    s=40,
                    zorder=3,
                    label="calendar-arb violation",
                )

        decay_axis.set_title(
            f"Snapshot {snapshot_id}: SVI time dynamics", loc="left"
        )
        decay_axis.set_ylabel(
            "IV drift per calendar day held\n(vol points, = -d(sigma)/d(tau)/365)"
        )
        decomposition_axis.set_ylabel(
            "Decomposition of that drift\n(vol points per day)"
        )
        calendar_axis.set_ylabel(
            "Calendar no-arb margin\n(w_far - w_near) / w_near (%)"
        )
        calendar_axis.set_xlabel("Log-moneyness k = ln(K / F)")

        _style_axes(axes)

        footer = (
            "d(sigma)/d(tau) = dw_dtau / (2 sigma tau) - sigma / (2 tau), from a "
            "linear-in-tau interpolation of total variance at fixed k (the "
            "interpolation that preserves calendar-arb freeness).\n"
            "Sign: positive on panel 1 means IV rises as you hold. A negative "
            "d(sigma)/d(tau) is NOT an arbitrage - the no-arb constraint binds on "
            "w, which is panel 3, and is checked only on the pairwise overlap.\n"
            "Same k is a different strike at each expiry, because F differs."
        )
        _save(figure, output, footer)


# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SVI spot and time dynamics for one snapshot."
    )
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--snapshot-id", type=int)
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--number-of-starts", type=int, default=16)
    parser.add_argument("--max-abs-k", type=float, default=0.75)
    parser.add_argument("--max-relative-spread", type=float)
    parser.add_argument(
        "--min-observations",
        type=int,
        default=DEFAULT_MINIMUM_OBSERVATIONS,
        help="Must exceed 5, the number of free SVI parameters.",
    )
    parser.add_argument(
        "--skew-stickiness-ratio",
        type=float,
        default=0.0,
        help="R. 0 sticky-delta, 1 sticky-strike, ~1.5 typical index.",
    )
    parser.add_argument(
        "--forward-spot-elasticity",
        type=float,
        default=1.0,
        help="eps = d lnF / d lnS.",
    )
    parser.add_argument(
        "--evaluate-at",
        choices=("near", "far", "midpoint"),
        default="near",
        help="Which tau the time derivative is evaluated at.",
    )
    parser.add_argument("--spot-output", default="figs/svi_spot_dynamics.png")
    parser.add_argument("--time-output", default="figs/svi_time_dynamics.png")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
        SurfaceFilterPolicy(max_relative_spread=args.max_relative_spread),
    )

    observations_by_expiry: dict[int, list] = defaultdict(list)
    for item in surface.observations:
        if abs(item.log_moneyness) <= args.max_abs_k:
            observations_by_expiry[item.expiration_timestamp].append(item)

    smiles = fit_snapshot_term_structure(
        observations_by_expiry,
        number_of_starts=args.number_of_starts,
        min_observations=args.min_observations,
    )
    if not smiles:
        raise SystemExit("No expiry survived calibration")

    plot_spot_dynamics(
        smiles,
        Path(args.spot_output),
        snapshot_id=snapshot_id,
        skew_stickiness_ratio=args.skew_stickiness_ratio,
        forward_spot_elasticity=args.forward_spot_elasticity,
    )

    # Compute the segments ONCE and use the same objects for both the figure
    # and the console report. The previous version called calendar_slope with
    # n_points=201 inside the plot and n_points=101 in __main__, so a narrow
    # violation could show as crosses on the figure while the console said OK.
    segments = build_time_derivative_segments(
        smiles, evaluate_at=args.evaluate_at
    )
    if segments:
        plot_time_dynamics(
            segments, Path(args.time_output), snapshot_id=snapshot_id
        )
    else:
        print("No adjacent pair shared a log-moneyness range; "
              "skipping time-dynamics figure.")

    print(f"snapshot={snapshot_id} fitted {len(smiles)} expiries")
    for smile in smiles:
        p = smile.parameters
        print(
            f"  expiry={smile.expiration_timestamp} tau={smile.tau:.4f} "
            f"n={smile.observation_count} "
            f"theta_atm={p.atm_total_variance():.6f} "
            f"left_wing={p.left_wing_slope:.4f} "
            f"right_wing={p.right_wing_slope:.4f} "
            f"k_min_var={p.minimum_total_variance_k:.4f} "
            f"curvature={p.atm_curvature:.3f}"
        )
    for segment in segments:
        status = "OK" if segment.calendar_arbitrage_free else "VIOLATION"
        print(
            f"  calendar {segment.expiry_near} -> {segment.expiry_far}: "
            f"min relative margin="
            f"{segment.minimum_calendar_margin_relative:.3e} ({status})"
        )
    print(f"figures: {args.spot_output}, {args.time_output}")