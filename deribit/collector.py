import argparse
import asyncio
from datetime import datetime, timezone
import logging
import signal
import sys
import time

from deribit.config import SnapshotUniversalConfig
from deribit.store import SnapshotStore
from deribit.ws_client import DeribitWSClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Collector")


class SnapshotCollectorDaemon:
    def __init__(
        self,
        db_path: str = "snapshots.db",
        currency: str = "BTC",
        interval_seconds: int = 900,  # 15 minutes
        prod: bool = True,
    ):
        self.db_path = db_path
        self.currency = currency.upper()
        self.interval = interval_seconds
        self.testnet = not prod
        self.store = SnapshotStore(db_path)
        self.config = SnapshotUniversalConfig(currency=self.currency)
        self.running = False

    async def capture_single_snapshot(self) -> int:
        client = DeribitWSClient(testnet=self.testnet)
        t_start = time.time_ns()
        try:
            await client.connect()
            raw_payload = await client.fetch_snapshot_data(self.config)
        finally:
            await client.close()

        latency_ms = (time.time_ns() - t_start) / 1_000_000.0
        snapshot_id = self.store.save_snapshot(self.currency, raw_payload)
        
        meta = self.store.load_snapshot_metadata(snapshot_id)
        index_px = raw_payload.get("index", {}).get("payload", {}).get("index_price", 0.0)
        num_options = len(raw_payload.get("options", {}).get("payload", []))
        num_futures = len(raw_payload.get("futures", {}).get("payload", []))
        skew_ms = meta.server_skew_ms if meta and meta.server_skew_ms else 0.0

        logger.info(
            f"Saved Snapshot #{snapshot_id:<4d} | {self.currency} Index: ${index_px:,.2f} | Options: {num_options:<3d} | Futures: {num_futures:<2d} | Skew: {skew_ms:.1f}ms | Latency: {latency_ms:.1f}ms"
        )
        return snapshot_id

    async def run(self):
        self.running = True
        env_label = "MAINNET (Production)" if not self.testnet else "TESTNET"
        logger.info("=" * 75)
        logger.info("PRISM Snapshot Collector started.")
        logger.info("Environment:  %s", env_label)
        logger.info("Currency:     %s", self.currency)
        logger.info("Interval:     %d seconds (%0.1f minutes)", self.interval, self.interval / 60.0)
        logger.info("Destination:  %s", self.db_path)
        logger.info("=" * 75)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except (NotImplementedError, AttributeError):
                pass

        while self.running:
            try:
                await self.capture_single_snapshot()
            except Exception as exc:
                logger.error("Snapshot capture failed: %s. Retrying in %ds...", exc, self.interval)

            # Sleep until next scheduled boundary
            await asyncio.sleep(self.interval)

    def stop(self):
        logger.info("Stopping collector daemon...")
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="PRISM Persistent Snapshot Collector Daemon")
    parser.add_argument("--db", default="snapshots.db", help="Path to SQLite database")
    parser.add_argument("--currency", default="BTC", help="Asset currency (BTC/ETH)")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds (default: 300s = 5m)")
    parser.add_argument("--prod", action="store_true", help="Capture from Deribit Production (mainnet)")
    args = parser.parse_args()

    daemon = SnapshotCollectorDaemon(
        db_path=args.db,
        currency=args.currency,
        interval_seconds=args.interval,
        prod=args.prod,
    )

    try:
        asyncio.run(daemon.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Collector shutdown complete.")


if __name__ == "__main__":
    main()