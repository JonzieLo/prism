import asyncio
import json
import os
import time
from typing import Any, Dict , Optional
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from .config import SnapshotUniversalConfig, build_jrpc_requests

load_dotenv()
WS_URL_TESTNET = "wss://test.deribit.com/ws/api/v2"
WS_URL_PROD = "wss://asia.deribit.com/ws/api/v2"


class DeribitWSClient:
    def __init__(self, testnet: Optional[bool] = True):
        if testnet is None:
            testnet = os.getenv("DERIBIT_TESTNET", "True").strip() == "True"
        self.url = WS_URL_TESTNET if testnet else WS_URL_PROD
        self.ws: Optional[ClientConnection] = None

    async def connect(self):
        if self.ws is None:
            self.ws = await websockets.connect(self.url)

    async def close(self):
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    async def fetch_snapshot_data(self, config: SnapshotUniversalConfig) -> Dict[str, Any]:
        if self.ws is None:
            await self.connect()

        requests = build_jrpc_requests(config)
        id_to_key = {req["id"]: key for key, req in requests.items()}

        try:
            sent_at = time.time_ns()
            for req in requests.values():
                await self.ws.send(json.dumps(req))

            responses = {}
            for _ in range(len(requests)):
                raw_frame = await self.ws.recv()
                received_at = time.time_ns()

                payload = json.loads(raw_frame)
                req_id = payload.get("id")

                if req_id in id_to_key:
                    key = id_to_key[req_id]
                    responses[key] = {
                        "payload": payload.get("result"),
                        "usIn": payload.get("usIn"),
                        "usOut": payload.get("usOut"),
                        "sent_at_ns": sent_at,
                        "received_at_ns": received_at,
                    }
            return responses
        except Exception as e:
            await self.close()
            raise e