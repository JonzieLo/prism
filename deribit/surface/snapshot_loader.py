"""Load surface observations for a snapshot, or for one expiry within it.

Both benchmark scripts had the same eight-line block inlined: load snapshot,
build the forward curve, build surface observations, filter on |k|, group by
expiry. Cross-snapshot work (measuring the backbone from two snapshots of the
same expiry) needs that block twice with different snapshot ids, which is the
point at which it has to become a function.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from deribit.forward_curve import build_forward_curve
from deribit.store import SnapshotStore

from .calibration import calibrate_svi_slice
from .filter_policy import SurfaceFilterPolicy
from .observations import SurfaceObservation
from .pipeline import build_surface_observations
from .results import CalibratedSmile

DEFAULT_MAX_ABS_K = 0.75


@dataclass(frozen=True)
class LoadedSnapshot:
    snapshot_id: int
    timestamp_ns: int | None
    observations_by_expiry: dict[int, list[SurfaceObservation]]

    @property
    def timestamp(self) -> datetime | None:
        if self.timestamp_ns is None:
            return None
        return datetime.fromtimestamp(
            self.timestamp_ns / 1_000_000_000.0, tz=timezone.utc
        )

    def label(self) -> str:
        stamp = self.timestamp
        when = (
            stamp.strftime("%Y-%m-%d %H:%M UTC")
            if stamp is not None
            else "timestamp unavailable"
        )
        return f"snapshot {self.snapshot_id} ({when})"

    def expiry(self, expiration_timestamp: int) -> list[SurfaceObservation]:
        items = self.observations_by_expiry.get(expiration_timestamp)
        if not items:
            available = ", ".join(
                str(key) for key in sorted(self.observations_by_expiry)
            )
            raise ValueError(
                f"Snapshot {self.snapshot_id} has no eligible observations "
                f"for expiry {expiration_timestamp}. Available: {available}"
            )
        return items


def load_snapshot_observations(
    store: SnapshotStore,
    snapshot_id: int,
    *,
    max_abs_k: float = DEFAULT_MAX_ABS_K,
    max_relative_spread: float | None = None,
) -> LoadedSnapshot:
    """Every eligible observation in one snapshot, grouped by expiry."""
    snapshot = store.load_snapshot(snapshot_id)
    if snapshot is None:
        raise ValueError(f"Snapshot {snapshot_id} does not exist")

    curve = build_forward_curve(snapshot)
    surface = build_surface_observations(
        snapshot_id,
        list(curve.quotes),
        list(curve.expiry_forwards),
        SurfaceFilterPolicy(max_relative_spread=max_relative_spread),
    )

    grouped: dict[int, list[SurfaceObservation]] = defaultdict(list)
    for item in surface.observations:
        if abs(item.log_moneyness) <= max_abs_k:
            grouped[item.expiration_timestamp].append(item)

    metadata = store.load_snapshot_metadata(snapshot_id)
    return LoadedSnapshot(
        snapshot_id=snapshot_id,
        timestamp_ns=(metadata.timestamp_ns if metadata is not None else None),
        observations_by_expiry=dict(grouped),
    )


def load_expiry_observations(
    store: SnapshotStore,
    snapshot_id: int,
    expiration_timestamp: int,
    *,
    max_abs_k: float = DEFAULT_MAX_ABS_K,
    max_relative_spread: float | None = None,
) -> list[SurfaceObservation]:
    """One expiry from one snapshot.

    This is the `obs_<date>_<expiry>` in the backbone worked example:

        obs = load_expiry_observations(store, 19, 1790323200000)
    """
    loaded = load_snapshot_observations(
        store,
        snapshot_id,
        max_abs_k=max_abs_k,
        max_relative_spread=max_relative_spread,
    )
    return loaded.expiry(expiration_timestamp)


def calibrate_expiry(
    store: SnapshotStore,
    snapshot_id: int,
    expiration_timestamp: int,
    *,
    max_abs_k: float = DEFAULT_MAX_ABS_K,
    max_relative_spread: float | None = None,
    number_of_starts: int = 16,
) -> CalibratedSmile:
    """Load and fit one expiry in one call."""
    return calibrate_svi_slice(
        load_expiry_observations(
            store,
            snapshot_id,
            expiration_timestamp,
            max_abs_k=max_abs_k,
            max_relative_spread=max_relative_spread,
        ),
        number_of_starts=number_of_starts,
    )


def find_snapshots_with_expiry(
    store: SnapshotStore,
    expiration_timestamp: int,
    *,
    snapshot_ids: list[int] | None = None,
    currency: str = "BTC",
    max_abs_k: float = DEFAULT_MAX_ABS_K,
    minimum_observations: int = 10,
) -> list[LoadedSnapshot]:
    """Which snapshots carry a usable chain for this expiry, oldest first.

    Answers "do I actually have two snapshots of the 25 Sep expiry" before
    any calibration is attempted.
    """
    if snapshot_ids is None:
        latest = store.latest_snapshot_id(currency)
        if latest is None:
            return []
        snapshot_ids = list(range(1, latest + 1))

    found: list[LoadedSnapshot] = []
    for snapshot_id in snapshot_ids:
        try:
            loaded = load_snapshot_observations(
                store, snapshot_id, max_abs_k=max_abs_k
            )
        except Exception:
            continue
        items = loaded.observations_by_expiry.get(expiration_timestamp, [])
        if len(items) >= minimum_observations:
            found.append(loaded)
    return sorted(found, key=lambda item: item.timestamp_ns or 0)
