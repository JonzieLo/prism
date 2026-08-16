import pytest
from deribit.config import SnapshotUniversalConfig
from deribit.ws_client import DeribitWSClient


@pytest.mark.asyncio
async def test_deribit_websocket_snapshot():
    """Verifies Deribit WebSocket transport, payload structure, and server timestamp skew."""
    client = DeribitWSClient(testnet=False)
    config = SnapshotUniversalConfig(currency="BTC")

    data = await client.fetch_snapshot_data(config)

    assert "index" in data
    assert "instruments" in data
    assert "options" in data
    assert "futures" in data

    assert data["index"]["payload"]["index_price"] > 0.0
    assert len(data["instruments"]["payload"]) > 0
    assert len(data["options"]["payload"]) > 0

    us_out_stamps = [resp["usOut"] for resp in data.values() if resp.get("usOut")]
    if us_out_stamps:
        skew_ms = (max(us_out_stamps) - min(us_out_stamps)) / 1000.0
        assert skew_ms < 100.0, f"Exchange server skew too high: {skew_ms:.2f} ms"