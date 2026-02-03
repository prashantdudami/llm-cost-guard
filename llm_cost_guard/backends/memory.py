"""
In-memory storage backend for LLM Cost Guard.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading

from llm_cost_guard.backends.base import Backend
from llm_cost_guard.models import CostRecord, CostReport


class MemoryBackend(Backend):
    """Thread-safe in-memory storage backend."""

    def __init__(self, max_records: int = 100000, **kwargs):
        """
        Initialize the memory backend.

        Args:
            max_records: Maximum number of records to keep in memory
        """
        self._records: List[CostRecord] = []
        self._max_records = max_records
        self._lock = threading.RLock()

    def save_record(self, record: CostRecord) -> None:
        """Save a cost record."""
        with self._lock:
            self._records.append(record)
            # Evict old records if we exceed the limit
            if len(self._records) > self._max_records:
                # Remove oldest 10% of records
                evict_count = self._max_records // 10
                self._records = self._records[evict_count:]

    def save_records(self, records: List[CostRecord]) -> None:
        """Save multiple cost records."""
        with self._lock:
            self._records.extend(records)
            # Evict old records if we exceed the limit
            if len(self._records) > self._max_records:
                evict_count = len(self._records) - self._max_records
                self._records = self._records[evict_count:]

    def _matches_filters(
        self,
        record: CostRecord,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        tags: Optional[Dict[str, str]],
    ) -> bool:
        """Check if a record matches the given filters."""
        if start_date and record.timestamp < start_date:
            return False
        if end_date and record.timestamp > end_date:
            return False
        if tags:
            for key, value in tags.items():
                if record.tags.get(key) != value:
                    return False
        return True

    def get_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[CostRecord]:
        """Retrieve cost records with optional filters."""
        with self._lock:
            filtered = [
                r for r in self._records if self._matches_filters(r, start_date, end_date, tags)
            ]

            # Sort by timestamp descending (most recent first)
            filtered.sort(key=lambda r: r.timestamp, reverse=True)

            # Apply offset and limit
            if offset:
                filtered = filtered[offset:]
            if limit:
                filtered = filtered[:limit]

            return filtered

    def get_total_cost(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get total cost for the given filters."""
        with self._lock:
            total = 0.0
            for record in self._records:
                if self._matches_filters(record, start_date, end_date, tags):
                    total += record.total_cost
            return total

    def get_aggregated_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        group_by: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get aggregated costs grouped by specified fields."""
        with self._lock:
            if not group_by:
                # Return overall totals
                total_cost = 0.0
                total_calls = 0
                total_input_tokens = 0
                total_output_tokens = 0

                for record in self._records:
                    if self._matches_filters(record, start_date, end_date, tags):
                        total_cost += record.total_cost
                        total_calls += 1
                        total_input_tokens += record.input_tokens
                        total_output_tokens += record.output_tokens

                return {
                    "total_cost": total_cost,
                    "total_calls": total_calls,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                }

            # Group by specified fields
            groups: Dict[tuple, Dict[str, Any]] = defaultdict(
                lambda: {
                    "cost": 0.0,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )

            for record in self._records:
                if not self._matches_filters(record, start_date, end_date, tags):
                    continue

                # Build group key
                key_parts = []
                for field in group_by:
                    if field == "provider":
                        key_parts.append(record.provider)
                    elif field == "model":
                        key_parts.append(record.model)
                    elif field in record.tags:
                        key_parts.append(record.tags[field])
                    else:
                        key_parts.append("unknown")

                key = tuple(key_parts)
                groups[key]["cost"] += record.total_cost
                groups[key]["calls"] += 1
                groups[key]["input_tokens"] += record.input_tokens
                groups[key]["output_tokens"] += record.output_tokens

            # Convert to list format
            result = []
            for key, data in groups.items():
                row = dict(zip(group_by, key))
                row.update(data)
                result.append(row)

            # Sort by cost descending
            result.sort(key=lambda x: x["cost"], reverse=True)

            return {"groups": result, "group_by": group_by}

    def get_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        group_by: Optional[List[str]] = None,
    ) -> CostReport:
        """Generate a cost report."""
        with self._lock:
            records = self.get_records(start_date, end_date, tags)

            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            successful_calls = 0
            failed_calls = 0
            cache_hits = 0
            cache_savings = 0.0

            for record in records:
                total_cost += record.total_cost
                total_input_tokens += record.input_tokens
                total_output_tokens += record.output_tokens

                if record.success:
                    successful_calls += 1
                else:
                    failed_calls += 1

                if record.cached:
                    cache_hits += 1
                    cache_savings += record.cache_savings

            grouped_data = {}
            if group_by:
                agg = self.get_aggregated_costs(start_date, end_date, tags, group_by)
                grouped_data = agg.get("groups", [])

            return CostReport(
                start_date=start_date,
                end_date=end_date,
                total_cost=total_cost,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_calls=len(records),
                successful_calls=successful_calls,
                failed_calls=failed_calls,
                cache_hits=cache_hits,
                cache_savings=cache_savings,
                effective_cost=total_cost - cache_savings,
                records=records,
                grouped_data={"groups": grouped_data} if grouped_data else {},
            )

    def delete_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> int:
        """Delete records matching the filters."""
        with self._lock:
            initial_count = len(self._records)
            self._records = [
                r
                for r in self._records
                if not self._matches_filters(r, start_date, end_date, tags)
            ]
            return initial_count - len(self._records)

    def health_check(self) -> bool:
        """Check if the backend is healthy."""
        return True

    def close(self) -> None:
        """Close the backend (clears memory)."""
        with self._lock:
            self._records.clear()

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()

    @property
    def record_count(self) -> int:
        """Get the current number of records."""
        with self._lock:
            return len(self._records)

    def __enter__(self) -> "MemoryBackend":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
