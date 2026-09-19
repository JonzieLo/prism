
from datetime import datetime, timezone
from deribit.store import SnapshotStore
from deribit.forward_curve import build_forward_curve

snapshot_id = 20  # Using your newly fetched snapshot ID
store = SnapshotStore("snapshots.db")
snapshot = store.load_snapshot(snapshot_id)

if snapshot is None:
    raise SystemExit(f"Snapshot {snapshot_id} not found")

curve = build_forward_curve(snapshot)

print(f"--- Available Expiries for Snapshot {snapshot_id} ---")
for expiry in curve.expiry_forwards:
    date = datetime.fromtimestamp(expiry.expiration_timestamp / 1000, tz=timezone.utc)
    print(
        f"Expiry: {expiry.expiration_timestamp} | "
        f"Date: {date.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Pairs: {expiry.pair_count} | "
        f"Forward: {expiry.implied_forward:,.2f}"
    )
PY