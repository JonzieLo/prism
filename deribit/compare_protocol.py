import asyncio
import time
import httpx
from .config import SnapshotUniversalConfig
from .ws_client import DeribitWSClient

REST_URL = "https://test.deribit.com/api/v2"

async def fetch_rest_snapshot(cfg: SnapshotUniversalConfig):
    """Fetches snapshot data via async HTTP REST."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        sent_at_ns = time.time_ns()
        ccy = cfg.currency.upper()

        req_index = client.get(f"{REST_URL}/public/get_index_price?index_name={ccy.lower()}_usd")
        req_inst = client.get(f"{REST_URL}/public/get_instruments?currency={ccy}&kind=option&expired=false")
        req_opts = client.get(f"{REST_URL}/public/get_book_summary_by_currency?currency={ccy}&kind=option")
        req_futs = client.get(f"{REST_URL}/public/get_book_summary_by_currency?currency={ccy}&kind=future")

        res_index, res_inst, res_opts, res_futs = await asyncio.gather(
            req_index, req_inst, req_opts, req_futs
        )
        received_at_ns = time.time_ns()

        responses = {
            "index": res_index.json(),
            "instruments": res_inst.json(),
            "options": res_opts.json(),
            "futures": res_futs.json(),
        }

        # Wall-clock and exchange-side usOut skew
        wall_window_ms = (received_at_ns - sent_at_ns) / 1e6
        us_out_stamps = [r.get("usOut") for r in responses.values() if r.get("usOut")]
        server_skew_ms = (max(us_out_stamps) - min(us_out_stamps)) / 1000.0 if us_out_stamps else None

        return wall_window_ms, server_skew_ms


async def main():
    cfg = SnapshotUniversalConfig(currency="BTC")
    print(f"=== Transport Comparison Benchmark (M0) [{cfg.currency}]===\n")

    print("Testing HTTP REST transport...")
    rest_wall, rest_skew = await fetch_rest_snapshot(cfg)

    print("Testing WebSocket transport...")
    ws_client = DeribitWSClient(testnet=True)
    t_warm_start = time.time_ns()
    await ws_client.connect()
    await ws_client.fetch_snapshot_data(cfg) # Warm up run
    t_warm_end = time.time_ns()
    ws_warmup = (t_warm_end - t_warm_start) / 1e6

    t0 = time.time_ns()
    ws_data = await ws_client.fetch_snapshot_data(cfg)
    t1 = time.time_ns()

    await ws_client.close()
    
    ws_wall = (t1 - t0) / 1e6
    ws_stamps = [resp["usOut"] for resp in ws_data.values() if resp.get("usOut")]
    ws_skew = (max(ws_stamps) - min(ws_stamps)) / 1000.0 if ws_stamps else None

    speedup = ws_warmup / ws_wall if ws_wall > 0 else 0

    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'HTTP REST':<12} | {'WebSocket':<12}")
    print("="*50)
    print(f"{'Initial Warm-Up Wall (ms)':<32} | {'N/A':<12} | {ws_warmup:<12.2f}")
    print(f"{'Warmed-Up Pipe Wall (ms)':<32} | {rest_wall:<12.2f} | {ws_wall:<12.2f}")
    print(f"{'Exchange Server Skew (ms)':<32} | {rest_skew:<12.2f} | {ws_skew:<12.2f}")
    print("=" * 50)
    print(f"Persistent Websocket is {speedup:.1f}x faster than cold start")

if __name__ == "__main__":
    asyncio.run(main())