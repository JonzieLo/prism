import pytest
from deribit.config import SnapshotUniversalConfig
from deribit.store import SnapshotStore
from deribit.ws_client import DeribitWSClient


@pytest.mark.asyncio
async def test_snapshot_store_roundtrip():
    """Fetches live snapshot, saves to SQLite, and verifies byte-identical reloading."""
    config = SnapshotUniversalConfig(currency="BTC")
    client = DeribitWSClient(testnet=True)
    store = SnapshotStore("snapshots.db")

    live_data = await client.fetch_snapshot_data(config)
    await client.close()

    snapshot_id = store.save_snapshot(config.currency, live_data)
    reloaded_data = store.load_snapshot(snapshot_id)

    assert live_data == reloaded_data, "Error: Reloaded snapshot is not byte-identical."