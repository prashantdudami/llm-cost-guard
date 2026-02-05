"""
SQLite storage backend for LLM Cost Guard.
"""

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional

from llm_cost_guard.backends.base import Backend
from llm_cost_guard.models import CostRecord, CostReport, ModelType


class SQLiteBackend(Backend):
    """SQLite storage backend with thread-safe connection management."""

    def __init__(self, db_url: str, **kwargs):
        """
        Initialize the SQLite backend.

        Args:
            db_url: SQLite database URL (e.g., "sqlite:///costs.db" or "sqlite:///:memory:")
        """
        # Parse database path from URL
        if db_url.startswith("sqlite:///"):
            self._db_path = db_url[10:]
        elif db_url.startswith("sqlite://"):
            self._db_path = db_url[9:]
        else:
            self._db_path = db_url

        # Handle in-memory database
        if self._db_path == ":memory:" or self._db_path == "":
            self._db_path = ":memory:"

        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_database(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                model_type TEXT NOT NULL DEFAULT 'chat',
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                input_cost REAL NOT NULL DEFAULT 0.0,
                output_cost REAL NOT NULL DEFAULT 0.0,
                total_cost REAL NOT NULL DEFAULT 0.0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                tags TEXT NOT NULL DEFAULT '{}',
                metadata TEXT NOT NULL DEFAULT '{}',
                success INTEGER NOT NULL DEFAULT 1,
                error_type TEXT,
                cached INTEGER NOT NULL DEFAULT 0,
                cache_savings REAL NOT NULL DEFAULT 0.0,
                span_id TEXT
            )
        """
        )

        # Create indexes for common queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON cost_records(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_provider ON cost_records(provider)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON cost_records(model)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_span_id ON cost_records(span_id)"
        )

        conn.commit()

    def _record_to_row(self, record: CostRecord) -> tuple:
        """Convert a CostRecord to a database row tuple."""
        return (
            record.timestamp,
            record.provider,
            record.model,
            record.model_type.value,
            record.input_tokens,
            record.output_tokens,
            record.input_cost,
            record.output_cost,
            record.total_cost,
            record.latency_ms,
            json.dumps(record.tags),
            json.dumps(record.metadata),
            1 if record.success else 0,
            record.error_type,
            1 if record.cached else 0,
            record.cache_savings,
            record.span_id,
        )

    def _row_to_record(self, row: sqlite3.Row) -> CostRecord:
        """Convert a database row to a CostRecord."""
        return CostRecord(
            timestamp=row["timestamp"],
            provider=row["provider"],
            model=row["model"],
            model_type=ModelType(row["model_type"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            input_cost=row["input_cost"],
            output_cost=row["output_cost"],
            total_cost=row["total_cost"],
            latency_ms=row["latency_ms"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            success=bool(row["success"]),
            error_type=row["error_type"],
            cached=bool(row["cached"]),
            cache_savings=row["cache_savings"],
            span_id=row["span_id"],
        )

    def save_record(self, record: CostRecord) -> None:
        """Save a cost record."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO cost_records (
                timestamp, provider, model, model_type,
                input_tokens, output_tokens, input_cost, output_cost, total_cost,
                latency_ms, tags, metadata, success, error_type,
                cached, cache_savings, span_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            self._record_to_row(record),
        )

        conn.commit()

    def save_records(self, records: list[CostRecord]) -> None:
        """Save multiple cost records."""
        if not records:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.executemany(
            """
            INSERT INTO cost_records (
                timestamp, provider, model, model_type,
                input_tokens, output_tokens, input_cost, output_cost, total_cost,
                latency_ms, tags, metadata, success, error_type,
                cached, cache_savings, span_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [self._record_to_row(r) for r in records],
        )

        conn.commit()

    def _build_where_clause(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> tuple[str, list]:
        """Build WHERE clause and parameters."""
        conditions = []
        params = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if tags:
            for key, value in tags.items():
                # Security: Validate tag key to prevent SQL injection
                if not self._is_valid_identifier(key):
                    raise ValueError(f"Invalid tag key: {key}")
                # Use JSON extraction for tag filtering with validated key
                conditions.append(f"json_extract(tags, '$.{key}') = ?")
                params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params

    @staticmethod
    def _is_valid_identifier(name: str) -> bool:
        """
        Validate that a name is safe for use in SQL queries.

        Security: Prevents SQL injection via tag keys.
        """
        import re
        # Only allow alphanumeric, underscore, hyphen
        # Must start with letter or underscore
        if not name or len(name) > 128:
            return False
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_\-]*$', name))

    def get_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[dict[str, str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[CostRecord]:
        """Retrieve cost records with optional filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clause, params = self._build_where_clause(start_date, end_date, tags)

        query = f"""
            SELECT * FROM cost_records
            WHERE {where_clause}
            ORDER BY timestamp DESC
        """

        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_total_cost(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> float:
        """Get total cost for the given filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clause, params = self._build_where_clause(start_date, end_date, tags)

        cursor.execute(
            f"SELECT COALESCE(SUM(total_cost), 0) FROM cost_records WHERE {where_clause}",
            params,
        )

        result = cursor.fetchone()
        return float(result[0]) if result else 0.0

    def get_aggregated_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[dict[str, str]] = None,
        group_by: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get aggregated costs grouped by specified fields."""
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clause, params = self._build_where_clause(start_date, end_date, tags)

        if not group_by:
            cursor.execute(
                f"""
                SELECT
                    COALESCE(SUM(total_cost), 0) as total_cost,
                    COUNT(*) as total_calls,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens
                FROM cost_records
                WHERE {where_clause}
            """,
                params,
            )

            row = cursor.fetchone()
            return {
                "total_cost": float(row[0]),
                "total_calls": int(row[1]),
                "total_input_tokens": int(row[2]),
                "total_output_tokens": int(row[3]),
            }

        # Build GROUP BY clause
        group_columns = []
        for field in group_by:
            if field in ("provider", "model"):
                group_columns.append(field)
            else:
                # Assume it's a tag - validate to prevent SQL injection
                if not self._is_valid_identifier(field):
                    raise ValueError(f"Invalid group_by field: {field}")
                group_columns.append(f"json_extract(tags, '$.{field}') as {field}")

        ", ".join(
            [f if " as " not in f else f.split(" as ")[1] for f in group_columns]
        )
        group_cols = ", ".join(
            [f.split(" as ")[0] if " as " in f else f for f in group_columns]
        )

        cursor.execute(
            f"""
            SELECT
                {', '.join(group_columns)},
                SUM(total_cost) as cost,
                COUNT(*) as calls,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens
            FROM cost_records
            WHERE {where_clause}
            GROUP BY {group_cols}
            ORDER BY cost DESC
        """,
            params,
        )

        rows = cursor.fetchall()
        groups = []
        for row in rows:
            group_data = {}
            for i, field in enumerate(group_by):
                group_data[field] = row[i]
            group_data["cost"] = float(row[len(group_by)])
            group_data["calls"] = int(row[len(group_by) + 1])
            group_data["input_tokens"] = int(row[len(group_by) + 2])
            group_data["output_tokens"] = int(row[len(group_by) + 3])
            groups.append(group_data)

        return {"groups": groups, "group_by": group_by}

    def get_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[dict[str, str]] = None,
        group_by: Optional[list[str]] = None,
    ) -> CostReport:
        """Generate a cost report."""
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clause, params = self._build_where_clause(start_date, end_date, tags)

        cursor.execute(
            f"""
            SELECT
                COALESCE(SUM(total_cost), 0) as total_cost,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COUNT(*) as total_calls,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls,
                SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END) as cache_hits,
                COALESCE(SUM(cache_savings), 0) as cache_savings
            FROM cost_records
            WHERE {where_clause}
        """,
            params,
        )

        row = cursor.fetchone()

        grouped_data = {}
        if group_by:
            agg = self.get_aggregated_costs(start_date, end_date, tags, group_by)
            grouped_data = {"groups": agg.get("groups", [])}

        total_cost = float(row[0])
        cache_savings = float(row[7])

        return CostReport(
            start_date=start_date,
            end_date=end_date,
            total_cost=total_cost,
            total_input_tokens=int(row[1]),
            total_output_tokens=int(row[2]),
            total_calls=int(row[3]),
            successful_calls=int(row[4]),
            failed_calls=int(row[5]),
            cache_hits=int(row[6]),
            cache_savings=cache_savings,
            effective_cost=total_cost - cache_savings,
            records=[],  # Don't include all records in report for performance
            grouped_data=grouped_data,
        )

    def delete_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> int:
        """Delete records matching the filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clause, params = self._build_where_clause(start_date, end_date, tags)

        cursor.execute(f"DELETE FROM cost_records WHERE {where_clause}", params)
        deleted = cursor.rowcount

        conn.commit()
        return deleted

    def health_check(self) -> bool:
        """Check if the backend is healthy."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self._local.conn = None

    def __del__(self) -> None:
        """Destructor to ensure connection is closed."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def __enter__(self) -> "SQLiteBackend":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
