"""Measure the realised skew-stickiness ratio R from two snapshots of one expiry.

Every dvol/dspot number in the dynamics figure is conditional on an assumed R.
R = 0 (sticky delta) says fixed-strike vols move by the full skew; R = 1
(sticky strike) says they do not move at all. That is the whole width of the
answer, and assuming it makes the panel a restatement of the skew panel.

Two snapshots of the same expiry give one realised spot move and one realised
fixed-strike vol response, which is enough to back R out -- but only after
subtracting the time component, and only as an identification under a stated
d(w)/d(tau). With one spot move there is no error bar. Twenty-plus snapshots
and a regression of fixed-strike vol changes on d(lnS) and d(tau) turn this
into an estimate.

    python -m benchmarks.svi_backbone --early-snapshot 19 --late-snapshot 20 \
        --expiry 1790323200000
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deribit.store import SnapshotStore
from deribit.surface.calibration import calibrate_svi_slice
from deribit.surface.snapshot_loader import (
    find_snapshots_with_expiry,
    load_snapshot_observations,
)
from deribit.surface.term_structure import (
    DEFAULT_MINIMUM_OBSERVATIONS,
    dw_dtau_between,
    fit_snapshot_term_structure,
    measure_backbone,
)

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


def _utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1000.0, tz=timezone.utc
    ).strftime("%d %b %Y")


def plot_backbone(result, output: Path, *, early_label: str, late_label: str) -> None:
    arrays = {
        name: np.array([getattr(p, name) for p in result.points])
        for name in (
            "log_moneyness_early",
            "dvol_observed",
            "dvol_time_component",
            "dvol_spot_component",
            "implied_stickiness_ratio",
            "vega_weight",
            "dvol_dk_early",
        )
    }
    k = arrays["log_moneyness_early"]

    # R(k) has a pole wherever the skew crosses zero. Drawing through it
    # produces a spike that dominates the axis and means nothing, so mask
    # the region where the denominator is a small fraction of its own range.
    skew = np.abs(arrays["dvol_dk_early"])
    identified = skew >= 0.15 * skew.max()
    ratio_curve = np.where(identified, arrays["implied_stickiness_ratio"], np.nan)

    with plt.rc_context(_RC):
        figure, (move_axis, ratio_axis) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, layout="constrained"
        )

        move_axis.plot(
            k, arrays["dvol_observed"] * 100.0,
            color="#28251D", linewidth=2.0,
            label="observed fixed-strike IV change",
        )
        move_axis.plot(
            k, arrays["dvol_time_component"] * 100.0,
            color="#B98A2E", linewidth=1.6, linestyle="--",
            label="attributed to time (from dw/dtau)",
        )
        move_axis.plot(
            k, arrays["dvol_spot_component"] * 100.0,
            color="#20808D", linewidth=1.8,
            label="residual, attributed to spot",
        )
        move_axis.set_ylabel("IV change (vol points)")
        move_axis.set_title(
            f"Realised backbone: {early_label} -> {late_label}, "
            f"{_utc(result.expiration_timestamp)} expiry",
            loc="left",
        )

        ratio_axis.plot(
            k, ratio_curve,
            color="#20808D", linewidth=1.8,
            label="implied R(k), where the skew identifies it",
        )
        ratio_axis.axhline(
            result.least_squares_stickiness_ratio,
            color="#A13544", linewidth=1.8, linestyle="--",
            label=(
                "vega-weighted least squares R = "
                f"{result.least_squares_stickiness_ratio:.3f}"
            ),
        )
        ratio_axis.axhline(
            result.vega_weighted_stickiness_ratio,
            color="#B98A2E", linewidth=1.2, linestyle=":",
            label=(
                "pointwise vega-weighted R = "
                f"{result.vega_weighted_stickiness_ratio:.3f}"
            ),
        )
        for level, name in ((0.0, "sticky delta"), (1.0, "sticky strike")):
            ratio_axis.axhline(
                level, color=_NEUTRAL, linewidth=0.9, linestyle=":",
            )
            ratio_axis.annotate(
                name, xy=(k[0], level), fontsize=7.5, color=_NEUTRAL,
                va="bottom",
            )
        finite = ratio_curve[np.isfinite(ratio_curve)]
        if finite.size:
            low, high = float(np.min(finite)), float(np.max(finite))
            pad = max(0.3, 0.15 * (high - low))
            ratio_axis.set_ylim(min(low, -0.3) - pad, max(high, 1.3) + pad)
        ratio_axis.set_ylabel("Implied skew-stickiness ratio R")
        ratio_axis.set_xlabel("Log-moneyness k = ln(K / F_early)")

        for axis in (move_axis, ratio_axis):
            axis.axhline(0.0, color=_NEUTRAL, linewidth=0.9)
            axis.axvline(0.0, color=_NEUTRAL, linewidth=0.9, linestyle=":")
            axis.grid(axis="y", color="#D4D1CA", linewidth=0.7)
            axis.grid(axis="x", visible=False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.legend(frameon=False, fontsize=8)

        figure.text(
            0.01, -0.06,
            f"d(lnS) = {result.d_ln_spot:+.4f}, d(tau) = {result.d_tau:+.4f}y. "
            f"{result.identification_note}\n"
            "R is not identified where the skew is flat, so that region is "
            "masked. Prefer the least-squares R: the pointwise average "
            "divides by the skew, which vanishes near the smile minimum, "
            "and on this fit that sits close to ATM where vega is largest.",
            fontsize=7.6, color=_NEUTRAL,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output, dpi=180, bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Back out the realised skew-stickiness ratio from two snapshots "
            "of the same expiry."
        )
    )
    parser.add_argument("--db", default="snapshots.db")
    parser.add_argument("--early-snapshot", type=int)
    parser.add_argument("--late-snapshot", type=int)
    parser.add_argument("--expiry", type=int, required=True)
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--max-abs-k", type=float, default=0.75)
    parser.add_argument("--max-relative-spread", type=float)
    parser.add_argument("--number-of-starts", type=int, default=16)
    parser.add_argument(
        "--forward-spot-elasticity", type=float, default=1.0,
        help="eps = d lnF / d lnS. Assumed, not measured.",
    )
    parser.add_argument(
        "--list-snapshots", action="store_true",
        help="List snapshots carrying this expiry, then exit.",
    )
    parser.add_argument("--output", default="figs/svi_backbone.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    store = SnapshotStore(args.db)

    if args.list_snapshots or args.early_snapshot is None or args.late_snapshot is None:
        found = find_snapshots_with_expiry(
            store,
            args.expiry,
            currency=args.currency,
            max_abs_k=args.max_abs_k,
        )
        print(f"Snapshots carrying expiry {args.expiry} ({_utc(args.expiry)}):")
        for loaded in found:
            count = len(loaded.observations_by_expiry[args.expiry])
            print(f"  {loaded.label()}  n={count}")
        if args.list_snapshots:
            raise SystemExit(0)
        raise SystemExit(
            "Pass --early-snapshot and --late-snapshot, chosen from above."
        )

    early_loaded = load_snapshot_observations(
        store, args.early_snapshot,
        max_abs_k=args.max_abs_k,
        max_relative_spread=args.max_relative_spread,
    )
    late_loaded = load_snapshot_observations(
        store, args.late_snapshot,
        max_abs_k=args.max_abs_k,
        max_relative_spread=args.max_relative_spread,
    )

    early = calibrate_svi_slice(
        early_loaded.expiry(args.expiry),
        number_of_starts=args.number_of_starts,
    )
    late = calibrate_svi_slice(
        late_loaded.expiry(args.expiry),
        number_of_starts=args.number_of_starts,
    )

    # dw/dtau from the EARLY snapshot's own term structure: this expiry and
    # the next one out. Passed as a callable so it is evaluated on
    # measure_backbone's fixed-strike grid rather than a segment's grid.
    early_smiles = fit_snapshot_term_structure(
        early_loaded.observations_by_expiry,
        number_of_starts=args.number_of_starts,
        min_observations=DEFAULT_MINIMUM_OBSERVATIONS,
    )
    neighbours = [s for s in early_smiles if s.tau > early.tau]
    dw_dtau = None
    if neighbours:
        neighbour = neighbours[0]
        dw_dtau = dw_dtau_between(early, neighbour)
        print(
            f"dw/dtau from early-snapshot segment "
            f"{_utc(early.expiration_timestamp)} -> "
            f"{_utc(neighbour.expiration_timestamp)} "
            f"(tau {early.tau:.4f} -> {neighbour.tau:.4f}y)"
        )
    else:
        print(
            "No longer expiry in the early snapshot; falling back to "
            "dw/dtau = 0, which biases R."
        )

    result = measure_backbone(
        early, late,
        dw_dtau=dw_dtau,
        forward_spot_elasticity=args.forward_spot_elasticity,
    )

    plot_backbone(
        result, Path(args.output),
        early_label=early_loaded.label(),
        late_label=late_loaded.label(),
    )

    print()
    print(f"expiry            {args.expiry} ({_utc(args.expiry)})")
    print(f"early             {early_loaded.label()}  F={early.forward:,.2f}  tau={early.tau:.4f}y")
    print(f"late              {late_loaded.label()}  F={late.forward:,.2f}  tau={late.tau:.4f}y")
    print(f"d(lnF)            {result.d_ln_spot:+.5f}  ({np.expm1(result.d_ln_spot) * 100:+.2f}%)")
    print(f"d(tau)            {result.d_tau:+.5f}y  ({result.d_tau * 365.0:+.2f} days)")
    print(f"R (least squares) {result.least_squares_stickiness_ratio:.4f}   <- report this one")
    print(f"R (pointwise avg) {result.vega_weighted_stickiness_ratio:.4f}")
    print()

    # Conditioning. R is recovered by DIVIDING by d(lnS), so a small spot
    # move amplifies every other error without the arithmetic complaining.
    # measure_backbone only rejects |d(lnS)| < 1e-6, which is far short of
    # the move you need for the answer to mean anything: on this dataset a
    # 0.03% move returns R = -6 while a 1.3% move returns R = 0.93.
    warnings = []
    if abs(result.d_ln_spot) < 0.005:
        warnings.append(
            f"spot move is only {np.expm1(abs(result.d_ln_spot)) * 100:.2f}%; "
            "R is dominated by calibration noise below roughly 0.5%"
        )
    decay_days = abs(result.d_tau) * 365.0
    if decay_days > 2.0:
        warnings.append(
            f"snapshots are {decay_days:.1f} days apart; the smile has had "
            "time to reprice for reasons that are neither spot nor decay, "
            "and the dw/dtau correction is a secant extrapolated to fit"
        )
    if warnings:
        print("POORLY CONDITIONED -- do not report this R:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    print(result.identification_note)
    print(f"figure saved to: {args.output}")
