import json
import sqlite3
import time
from typing import Any, Dict, Optional

class SnapshotStore:
    def __init__(self, db_path: str = "snapshots.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshot_meta (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    currency TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    server_skew_ms REAL,
                    schema_version TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshot_data(
                    snapshot_id INTEGER PRIMARY KEY,
                    raw_json TEXT NOT NULL,
                    FOREIGN KEY(snapshot_id) REFERENCES snapshot_meta(snapshot_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS instrument_specs (
                    instrument_name TEXT PRIMARY KEY,
                    currency TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    settlement_currency TEXT NOT NULL,
                    expiration_timestamp INTEGER NOT NULL,
                    strike REAL,
                    option_type TEXT,
                    tick_size REAL NOT NULL,
                    contract_size REAL NOT NULL,
                    min_trade_amount REAL NOT NULL,
                    maker_commission REAL,
                    taker_commission REAL
                )
            """)
            conn.commit()

    def save_snapshot(self, currency: str, raw_responses: Dict[str, Any], schema_ver: str = "1.0") -> int:
        now_ns = time.time_ns()

        us_out_stamps = [resp["usOut"] for resp in raw_responses.values() if resp.get("usOut")]
        skew_ms = (max(us_out_stamps) - min(us_out_stamps)) / 1000.0 if us_out_stamps else 0.0

        with self._get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO snapshot_meta (currency, timestamp_ns, server_skew_ms, schema_version)
                VALUES (?, ?, ?, ?)
                """,
                (currency.upper(), now_ns, skew_ms, schema_ver),
            )
            snapshot_id = cursor.lastrowid

            cursor.execute(
                """
                INSERT INTO snapshot_data (snapshot_id, raw_json)
                VALUES (?, ?)
                """,
                (snapshot_id, json.dumps(raw_responses)),
            )

            instruments = raw_responses.get("instruments", {}).get("payload", [])
            for inst in instruments:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO instrument_specs (
                        instrument_name, 
                        currency,
                        kind,
                        settlement_currency,
                        expiration_timestamp,
                        strike,
                        option_type,
                        tick_size,
                        contract_size,
                        min_trade_amount,
                        maker_commission,
                        taker_commission
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        inst.get("instrument_name"),
                        inst.get("base_currency"),
                        inst.get("kind"),
                        inst.get("settlement_currency"),
                        inst.get("expiration_timestamp"),
                        inst.get("strike"),
                        inst.get("option_type"),
                        inst.get("tick_size"),
                        inst.get("contract_size"),
                        inst.get("min_trade_amount"),
                        inst.get("maker_commission"),
                        inst.get("taker_commission"),
                    ),
                )

            conn.commit()
            return snapshot_id

    def load_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_json FROM snapshot_data WHERE snapshot_id = ?", (snapshot_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None