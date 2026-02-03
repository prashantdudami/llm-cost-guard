"""
Redis backend for LLM Cost Guard with distributed budget enforcement.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from llm_cost_guard.backends.base import Backend
from llm_cost_guard.models import CostRecord, CostReport, ModelType

logger = logging.getLogger(__name__)

# Lua script for atomic budget check and reservation
BUDGET_CHECK_SCRIPT = """
local budget_key = KEYS[1]
local period_key = KEYS[2]
local amount = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local period_seconds = tonumber(ARGV[3])
local warning_threshold = tonumber(ARGV[4])

-- Get current spending
local current = tonumber(redis.call('GET', budget_key) or '0')

-- Check if we'd exceed the limit
local new_total = current + amount
if new_total > limit then
    return {-1, current, limit}  -- Exceeded
end

-- Check if we're at warning threshold
local warning = 0
if new_total >= (limit * warning_threshold) then
    warning = 1
end

-- Atomically increment spending
redis.call('INCRBYFLOAT', budget_key, amount)

-- Set expiry if not set (for period reset)
local ttl = redis.call('TTL', budget_key)
if ttl == -1 then
    redis.call('EXPIRE', budget_key, period_seconds)
end

return {new_total, current, warning}
"""

# Lua script for atomic budget reservation (pessimistic)
BUDGET_RESERVE_SCRIPT = """
local budget_key = KEYS[1]
local reservation_key = KEYS[2]
local amount = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local reservation_id = ARGV[3]
local period_seconds = tonumber(ARGV[4])

-- Get current spending + active reservations
local current = tonumber(redis.call('GET', budget_key) or '0')
local reserved = tonumber(redis.call('GET', reservation_key) or '0')
local effective = current + reserved

-- Check if we'd exceed the limit
if effective + amount > limit then
    return {-1, effective, limit}  -- Would exceed
end

-- Add to reservations
redis.call('INCRBYFLOAT', reservation_key, amount)
redis.call('EXPIRE', reservation_key, 300)  -- 5 minute reservation timeout

-- Store individual reservation for cleanup
redis.call('SETEX', 'reservation:' .. reservation_id, 300, amount)

return {effective + amount, effective, 0}  -- Success
"""

# Lua script for finalizing reservation
BUDGET_FINALIZE_SCRIPT = """
local budget_key = KEYS[1]
local reservation_key = KEYS[2]
local reserved_amount = tonumber(ARGV[1])
local actual_amount = tonumber(ARGV[2])
local reservation_id = ARGV[3]
local period_seconds = tonumber(ARGV[4])

-- Remove from reservations
redis.call('INCRBYFLOAT', reservation_key, -reserved_amount)

-- Add actual amount to spending
redis.call('INCRBYFLOAT', budget_key, actual_amount)

-- Set expiry for period reset
local ttl = redis.call('TTL', budget_key)
if ttl == -1 then
    redis.call('EXPIRE', budget_key, period_seconds)
end

-- Clean up reservation record
redis.call('DEL', 'reservation:' .. reservation_id)

return redis.call('GET', budget_key)
"""


class RedisBackend(Backend):
    """
    Redis backend with distributed budget enforcement.
    
    Features:
    - Atomic budget checks using Lua scripts
    - Pessimistic reservation for distributed consistency
    - Automatic period reset via TTL
    - Cost record storage with configurable retention
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "llm_cost_guard:",
        retention_days: int = 90,
        **kwargs: Any,
    ):
        """
        Initialize Redis backend.
        
        Args:
            url: Redis connection URL
            prefix: Key prefix for all Redis keys
            retention_days: How long to retain cost records
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package is required for Redis backend. "
                "Install with: pip install llm-cost-guard[redis]"
            )

        self._prefix = prefix
        self._retention_days = retention_days
        
        # Parse URL and connect
        self._client = redis.from_url(url, decode_responses=True, **kwargs)
        
        # Register Lua scripts
        self._budget_check_script = self._client.register_script(BUDGET_CHECK_SCRIPT)
        self._budget_reserve_script = self._client.register_script(BUDGET_RESERVE_SCRIPT)
        self._budget_finalize_script = self._client.register_script(BUDGET_FINALIZE_SCRIPT)
        
        # Metrics for graceful degradation
        self._metrics = {
            "backend_failures": 0,
            "fallback_activations": 0,
            "records_pending_sync": 0,
        }

    def _key(self, *parts: str) -> str:
        """Generate a prefixed key."""
        return self._prefix + ":".join(parts)

    # =========================================================================
    # Distributed Budget Enforcement
    # =========================================================================

    def check_budget_atomic(
        self,
        budget_name: str,
        amount: float,
        limit: float,
        period_seconds: int,
        warning_threshold: float = 0.8,
    ) -> Tuple[bool, float, bool]:
        """
        Atomically check and record spending against a budget.
        
        Args:
            budget_name: Name of the budget
            amount: Amount to add
            limit: Budget limit
            period_seconds: Period duration in seconds
            warning_threshold: Threshold for warning (0-1)
            
        Returns:
            Tuple of (allowed, current_spending, is_warning)
        """
        budget_key = self._key("budget", budget_name)
        period_key = self._key("budget_period", budget_name)
        
        try:
            result = self._budget_check_script(
                keys=[budget_key, period_key],
                args=[amount, limit, period_seconds, warning_threshold],
            )
            
            new_total, current, warning = result
            
            if new_total == -1:
                return False, current, False  # Exceeded
            
            return True, new_total, warning == 1
            
        except Exception as e:
            logger.error(f"Redis budget check failed: {e}")
            self._metrics["backend_failures"] += 1
            raise

    def reserve_budget(
        self,
        budget_name: str,
        estimated_amount: float,
        limit: float,
        reservation_id: str,
        period_seconds: int,
    ) -> Tuple[bool, float]:
        """
        Reserve budget before making an LLM call (pessimistic locking).
        
        Args:
            budget_name: Name of the budget
            estimated_amount: Estimated cost to reserve
            limit: Budget limit
            reservation_id: Unique ID for this reservation
            period_seconds: Period duration in seconds
            
        Returns:
            Tuple of (allowed, effective_spending)
        """
        budget_key = self._key("budget", budget_name)
        reservation_key = self._key("budget_reserved", budget_name)
        
        try:
            result = self._budget_reserve_script(
                keys=[budget_key, reservation_key],
                args=[estimated_amount, limit, reservation_id, period_seconds],
            )
            
            new_effective, current, _ = result
            
            if new_effective == -1:
                return False, current  # Would exceed
            
            return True, new_effective
            
        except Exception as e:
            logger.error(f"Redis budget reservation failed: {e}")
            self._metrics["backend_failures"] += 1
            raise

    def finalize_budget(
        self,
        budget_name: str,
        reserved_amount: float,
        actual_amount: float,
        reservation_id: str,
        period_seconds: int,
    ) -> float:
        """
        Finalize a budget reservation with actual cost.
        
        Args:
            budget_name: Name of the budget
            reserved_amount: Originally reserved amount
            actual_amount: Actual cost incurred
            reservation_id: Reservation ID
            period_seconds: Period duration in seconds
            
        Returns:
            New total spending
        """
        budget_key = self._key("budget", budget_name)
        reservation_key = self._key("budget_reserved", budget_name)
        
        try:
            result = self._budget_finalize_script(
                keys=[budget_key, reservation_key],
                args=[reserved_amount, actual_amount, reservation_id, period_seconds],
            )
            return float(result)
            
        except Exception as e:
            logger.error(f"Redis budget finalization failed: {e}")
            self._metrics["backend_failures"] += 1
            raise

    def release_reservation(
        self,
        budget_name: str,
        reserved_amount: float,
        reservation_id: str,
    ) -> None:
        """
        Release a reservation (on failure or cancellation).
        
        Args:
            budget_name: Name of the budget
            reserved_amount: Amount that was reserved
            reservation_id: Reservation ID
        """
        reservation_key = self._key("budget_reserved", budget_name)
        
        try:
            pipe = self._client.pipeline()
            pipe.incrbyfloat(reservation_key, -reserved_amount)
            pipe.delete(f"reservation:{reservation_id}")
            pipe.execute()
        except Exception as e:
            logger.error(f"Redis reservation release failed: {e}")
            self._metrics["backend_failures"] += 1

    def get_budget_spending(self, budget_name: str) -> float:
        """Get current spending for a budget."""
        budget_key = self._key("budget", budget_name)
        try:
            value = self._client.get(budget_key)
            return float(value) if value else 0.0
        except Exception as e:
            logger.error(f"Redis get budget spending failed: {e}")
            self._metrics["backend_failures"] += 1
            return 0.0

    def reset_budget(self, budget_name: str) -> None:
        """Reset a budget (for testing or manual reset)."""
        budget_key = self._key("budget", budget_name)
        reservation_key = self._key("budget_reserved", budget_name)
        try:
            self._client.delete(budget_key, reservation_key)
        except Exception as e:
            logger.error(f"Redis budget reset failed: {e}")
            self._metrics["backend_failures"] += 1

    # =========================================================================
    # Cost Record Storage
    # =========================================================================

    def save_record(self, record: CostRecord) -> None:
        """Save a cost record."""
        record_key = self._key("record", record.timestamp.strftime("%Y%m%d%H%M%S%f"))
        record_data = self._serialize_record(record)
        
        try:
            pipe = self._client.pipeline()
            
            # Save record with TTL
            ttl_seconds = self._retention_days * 24 * 60 * 60
            pipe.setex(record_key, ttl_seconds, json.dumps(record_data))
            
            # Add to sorted set for range queries (score = timestamp)
            records_key = self._key("records")
            score = record.timestamp.timestamp()
            pipe.zadd(records_key, {record_key: score})
            
            # Update aggregates for quick reporting
            self._update_aggregates(pipe, record)
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Redis save record failed: {e}")
            self._metrics["backend_failures"] += 1
            raise

    def save_records(self, records: List[CostRecord]) -> None:
        """Save multiple cost records."""
        for record in records:
            self.save_record(record)

    def _update_aggregates(self, pipe: Any, record: CostRecord) -> None:
        """Update aggregate counters for quick reporting."""
        date_str = record.timestamp.strftime("%Y-%m-%d")
        hour_str = record.timestamp.strftime("%Y-%m-%d-%H")
        
        # Daily aggregates
        daily_key = self._key("agg", "daily", date_str)
        pipe.hincrbyfloat(daily_key, "total_cost", record.total_cost)
        pipe.hincrby(daily_key, "total_calls", 1)
        pipe.hincrby(daily_key, "input_tokens", record.input_tokens)
        pipe.hincrby(daily_key, "output_tokens", record.output_tokens)
        pipe.expire(daily_key, self._retention_days * 24 * 60 * 60)
        
        # Model aggregates
        model_key = self._key("agg", "model", date_str, record.model)
        pipe.hincrbyfloat(model_key, "total_cost", record.total_cost)
        pipe.hincrby(model_key, "total_calls", 1)
        pipe.expire(model_key, self._retention_days * 24 * 60 * 60)
        
        # Tag aggregates
        for tag_key, tag_value in record.tags.items():
            tag_agg_key = self._key("agg", "tag", date_str, tag_key, tag_value)
            pipe.hincrbyfloat(tag_agg_key, "total_cost", record.total_cost)
            pipe.hincrby(tag_agg_key, "total_calls", 1)
            pipe.expire(tag_agg_key, self._retention_days * 24 * 60 * 60)

    def get_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 1000,
    ) -> List[CostRecord]:
        """Get cost records with optional filtering."""
        records_key = self._key("records")
        
        try:
            # Get record keys from sorted set
            min_score = start_date.timestamp() if start_date else "-inf"
            max_score = end_date.timestamp() if end_date else "+inf"
            
            record_keys = self._client.zrangebyscore(
                records_key, min_score, max_score, start=0, num=limit
            )
            
            if not record_keys:
                return []
            
            # Fetch records
            pipe = self._client.pipeline()
            for key in record_keys:
                pipe.get(key)
            
            results = pipe.execute()
            
            records = []
            for data in results:
                if data:
                    record = self._deserialize_record(json.loads(data))
                    
                    # Filter by tags if specified
                    if tags:
                        if all(record.tags.get(k) == v for k, v in tags.items()):
                            records.append(record)
                    else:
                        records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Redis get records failed: {e}")
            self._metrics["backend_failures"] += 1
            return []

    def get_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        group_by: Optional[List[str]] = None,
    ) -> CostReport:
        """Get aggregated cost report."""
        records = self.get_records(start_date, end_date, tags)
        
        total_cost = sum(r.total_cost for r in records)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in records)
        
        return CostReport(
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_calls=len(records),
            records=records,
            start_date=start_date,
            end_date=end_date,
            grouped_data={} if not group_by else self._group_records(records, group_by),
        )

    def _group_records(
        self, records: List[CostRecord], group_by: List[str]
    ) -> Dict[str, Any]:
        """Group records by specified fields."""
        groups: Dict[str, Dict[str, float]] = {}
        
        for record in records:
            key_parts = []
            for field in group_by:
                if field == "model":
                    key_parts.append(record.model)
                elif field == "provider":
                    key_parts.append(record.provider)
                elif field.startswith("tag:"):
                    tag_name = field[4:]
                    key_parts.append(record.tags.get(tag_name, "unknown"))
                else:
                    key_parts.append(record.tags.get(field, "unknown"))
            
            key = "|".join(key_parts)
            
            if key not in groups:
                groups[key] = {"cost": 0.0, "calls": 0, "tokens": 0}
            
            groups[key]["cost"] += record.total_cost
            groups[key]["calls"] += 1
            groups[key]["tokens"] += record.input_tokens + record.output_tokens
        
        return groups

    def get_total_cost(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get total cost for the given filters."""
        records = self.get_records(start_date, end_date, tags)
        return sum(r.total_cost for r in records)

    def get_aggregated_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        group_by: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get aggregated costs grouped by specified fields."""
        records = self.get_records(start_date, end_date, tags)
        
        if not group_by:
            return {
                "total_cost": sum(r.total_cost for r in records),
                "total_calls": len(records),
                "total_tokens": sum(r.input_tokens + r.output_tokens for r in records),
            }
        
        groups = self._group_records(records, group_by)
        return {"groups": [{"key": k, **v} for k, v in groups.items()]}

    def delete_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> int:
        """Delete records matching the filters."""
        records_key = self._key("records")
        
        try:
            # Get record keys to delete
            min_score = start_date.timestamp() if start_date else "-inf"
            max_score = end_date.timestamp() if end_date else "+inf"
            
            record_keys = self._client.zrangebyscore(records_key, min_score, max_score)
            
            if not record_keys:
                return 0
            
            # If tags filter, we need to check each record
            if tags:
                keys_to_delete = []
                pipe = self._client.pipeline()
                for key in record_keys:
                    pipe.get(key)
                results = pipe.execute()
                
                for key, data in zip(record_keys, results):
                    if data:
                        record = self._deserialize_record(json.loads(data))
                        if all(record.tags.get(k) == v for k, v in tags.items()):
                            keys_to_delete.append(key)
                record_keys = keys_to_delete
            
            if not record_keys:
                return 0
            
            # Delete records and remove from sorted set
            pipe = self._client.pipeline()
            for key in record_keys:
                pipe.delete(key)
                pipe.zrem(records_key, key)
            pipe.execute()
            
            return len(record_keys)
            
        except Exception as e:
            logger.error(f"Redis delete records failed: {e}")
            self._metrics["backend_failures"] += 1
            return 0

    def _serialize_record(self, record: CostRecord) -> Dict[str, Any]:
        """Serialize a CostRecord to dict."""
        return {
            "timestamp": record.timestamp.isoformat(),
            "provider": record.provider,
            "model": record.model,
            "model_type": record.model_type.value if record.model_type else "chat",
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "input_cost": record.input_cost,
            "output_cost": record.output_cost,
            "total_cost": record.total_cost,
            "latency_ms": record.latency_ms,
            "tags": record.tags,
            "metadata": record.metadata,
            "success": record.success,
            "error_type": record.error_type,
            "cached": record.cached,
            "cache_savings": record.cache_savings,
            "span_id": record.span_id,
        }

    def _deserialize_record(self, data: Dict[str, Any]) -> CostRecord:
        """Deserialize a dict to CostRecord."""
        return CostRecord(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            provider=data["provider"],
            model=data["model"],
            model_type=ModelType(data.get("model_type", "chat")),
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            input_cost=data["input_cost"],
            output_cost=data["output_cost"],
            total_cost=data["total_cost"],
            latency_ms=data["latency_ms"],
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
            success=data.get("success", True),
            error_type=data.get("error_type"),
            cached=data.get("cached", False),
            cache_savings=data.get("cache_savings", 0.0),
            span_id=data.get("span_id"),
        )

    # =========================================================================
    # Health & Metrics
    # =========================================================================

    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics for observability."""
        return {
            **self._metrics,
            "connected": self.health_check(),
        }

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()
