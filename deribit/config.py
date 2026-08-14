from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class SnapshotUniversalConfig:
    currency: str = 'BTC'
    include_options: bool = True
    include_futures: bool = True
    include_index: bool = True

def build_jrpc_requests(config: SnapshotUniversalConfig, start_id: int = 1) -> Dict[str, Dict[str, Any]]:
    requests = {}
    current_id = start_id
    ccy = config.currency.upper()

    if config.include_options:
        requests['instruments'] = {
            "jsonrpc": "2.0",
            "id": current_id,
            "method": "public/get_instruments",
            "params": {"currency": ccy, "kind": "option", "expired": False},
        }
        current_id += 1

        requests["options"] = {
            "jsonrpc": "2.0",
            "id": current_id,
            "method": "public/get_book_summary_by_currency",
            "params": {"currency": ccy, "kind": "option"},
        }
        current_id += 1

    if config.include_futures:
        requests["futures"] = {
            "jsonrpc": "2.0",
            "id": current_id,
            "method": "public/get_book_summary_by_currency",
            "params": {"currency": ccy, "kind": "future"},
        }
        current_id += 1

    if config.include_index:
        requests["index"] = {
            "jsonrpc": "2.0",
            "id": current_id,
            "method": "public/get_index_price",
            "params": {"index_name": f"{ccy.lower()}_usd"},
        }
        current_id += 1

    return requests