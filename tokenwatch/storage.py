"""
SQLite-based persistent storage for LLM call tracking.
Thread-safe with proper connection handling.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".tokenwatch", "costs.db")

# Thread-local storage for SQLite connections
_local = threading.local()


class Storage:
    """
    Thread-safe SQLite storage for LLM cost tracking.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating it if necessary."""
        if not hasattr(_local, "connections"):
            _local.connections = {}
        if self.db_path not in _local.connections:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            _local.connections[self.db_path] = conn
        return _local.connections[self.db_path]

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                provider    TEXT    NOT NULL,
                model       TEXT    NOT NULL,
                input_tokens  INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd    REAL    NOT NULL DEFAULT 0.0,
                session_id  TEXT,
                metadata    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_calls_timestamp  ON calls(timestamp);
            CREATE INDEX IF NOT EXISTS idx_calls_provider   ON calls(provider);
            CREATE INDEX IF NOT EXISTS idx_calls_session_id ON calls(session_id);

            CREATE TABLE IF NOT EXISTS budgets (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                limit_usd  REAL    NOT NULL,
                period     TEXT    NOT NULL CHECK(period IN ('daily','monthly','session','total')),
                created_at TEXT    NOT NULL
            );
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Log a single LLM API call.

        Returns:
            The row id of the inserted record.
        """
        conn = self._get_conn()
        timestamp = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None

        cursor = conn.execute(
            """
            INSERT INTO calls
                (timestamp, provider, model, input_tokens, output_tokens, cost_usd, session_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, provider, model, input_tokens, output_tokens, cost, session_id, meta_json),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_spend(
        self,
        period: str = "daily",
        session_id: Optional[str] = None,
    ) -> float:
        """
        Return total spend for the given period.

        Args:
            period: One of "daily", "monthly", "session", "total"
            session_id: Required when period="session"

        Returns:
            Total cost in USD.
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc)

        if period == "daily":
            since = now.strftime("%Y-%m-%dT00:00:00")
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE timestamp >= ?",
                (since,),
            ).fetchone()
        elif period == "monthly":
            since = now.strftime("%Y-%m-01T00:00:00")
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE timestamp >= ?",
                (since,),
            ).fetchone()
        elif period == "session":
            if not session_id:
                return 0.0
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        elif period == "total":
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM calls"
            ).fetchone()
        else:
            raise ValueError(f"Unknown period: '{period}'. Use daily/monthly/session/total.")

        return float(row[0]) if row else 0.0

    def get_history(
        self,
        limit: int = 50,
        provider: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return recent call history.

        Args:
            limit: Maximum number of records to return.
            provider: Filter by provider name.
            session_id: Filter by session ID.

        Returns:
            List of dicts with call details.
        """
        conn = self._get_conn()
        conditions: List[str] = []
        params: List[Any] = []

        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        rows = conn.execute(
            f"SELECT * FROM calls {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            if d.get("metadata"):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result

    def get_spend_by_provider(
        self,
        period: str = "daily",
    ) -> Dict[str, float]:
        """Return spend broken down by provider for the given period."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc)

        if period == "daily":
            since = now.strftime("%Y-%m-%dT00:00:00")
            clause = "WHERE timestamp >= ?"
            params: tuple = (since,)
        elif period == "monthly":
            since = now.strftime("%Y-%m-01T00:00:00")
            clause = "WHERE timestamp >= ?"
            params = (since,)
        else:
            clause = ""
            params = ()

        rows = conn.execute(
            f"SELECT provider, COALESCE(SUM(cost_usd), 0) as total FROM calls {clause} GROUP BY provider",
            params,
        ).fetchall()

        return {row["provider"]: float(row["total"]) for row in rows}

    def get_call_count(
        self,
        period: str = "daily",
        session_id: Optional[str] = None,
    ) -> int:
        """Return number of calls for the given period."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc)

        if period == "daily":
            since = now.strftime("%Y-%m-%dT00:00:00")
            row = conn.execute(
                "SELECT COUNT(*) FROM calls WHERE timestamp >= ?",
                (since,),
            ).fetchone()
        elif period == "monthly":
            since = now.strftime("%Y-%m-01T00:00:00")
            row = conn.execute(
                "SELECT COUNT(*) FROM calls WHERE timestamp >= ?",
                (since,),
            ).fetchone()
        elif period == "session" and session_id:
            row = conn.execute(
                "SELECT COUNT(*) FROM calls WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM calls").fetchone()

        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, filepath: str) -> None:
        """Export all call history to a CSV file."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, timestamp, provider, model, input_tokens, output_tokens, "
            "cost_usd, session_id, metadata FROM calls ORDER BY timestamp DESC"
        ).fetchall()

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["id", "timestamp", "provider", "model", "input_tokens",
                 "output_tokens", "cost_usd", "session_id", "metadata"]
            )
            for row in rows:
                writer.writerow(list(row))

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> int:
        """Delete all records for a session. Returns number of deleted rows."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM calls WHERE session_id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount

    def clear_all(self) -> None:
        """Delete ALL call records. Use with caution."""
        conn = self._get_conn()
        conn.execute("DELETE FROM calls")
        conn.commit()

    def close(self) -> None:
        """Close the thread-local connection if open."""
        if hasattr(_local, "connections") and self.db_path in _local.connections:
            _local.connections[self.db_path].close()
            del _local.connections[self.db_path]
