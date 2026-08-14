import asyncio
from .config import SnapshotUniversalConfig
from .ws_client import DeribitWSClient

async def main():
    client = DeribitWSClient(testnet=True)
    config = SnapshotUniversalConfig(currency="ETH")

    print(f"Connecting to Deribit WebSocket for {config.currency}...")

    data = await client.fetch_snapshot_data(config)
    print(f"\n--- Captured WebSocket Snapshot ({config.currency}) ---")
    print(f"Index Price: ${data['index']['payload']['index_price']:,.2f}")
    print(f"Options Discovered: {len(data['instruments']['payload'])}")
    print(f"Option Summaries: {len(data['options']['payload'])}")
    print(f"Futures Summaries: {len(data['futures']['payload'])}")
    
    # Calculate exchange-side timestamp spread (skew) across the 4 responses
    us_out_stamps = [resp["usOut"] for resp in data.values() if resp.get("usOut")]
    if us_out_stamps:
        skew_ms = (max(us_out_stamps) - min(us_out_stamps)) / 1000.0
        print(f"Exchange Server Timestamp Skew: {skew_ms:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())