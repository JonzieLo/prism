import asyncio
from deribit.config import SnapshotUniversalConfig
from deribit.store import SnapshotStore
from deribit.ws_client import DeribitWSClient


async def main():
    config = SnapshotUniversalConfig(currency="BTC")
    client = DeribitWSClient(testnet=True)
    store = SnapshotStore("snapshots.db")

    print("Capturing live market snapshot...")
    live_data = await client.fetch_snapshot_data(config)
    await client.close()

    snapshot_id = store.save_snapshot(config.currency, live_data)
    print(f"Snapshot persisted to SQLite with ID: {snapshot_id}")

    reloaded_data = store.load_snapshot(snapshot_id)

    assert live_data == reloaded_data, "Error: Reloaded snapshot is not byte-identical!"
    print("Reloaded snapshot is byte-identical to live.")


if __name__ == "__main__":
    asyncio.run(main())