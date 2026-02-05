"""
Audit logging for LLM Cost Guard.

Provides compliance-ready audit trails for:
- Budget changes (created, modified, deleted)
- Budget exceeded events
- Configuration changes
- Rate limit events
"""

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Budget events
    BUDGET_CREATED = "budget.created"
    BUDGET_MODIFIED = "budget.modified"
    BUDGET_DELETED = "budget.deleted"
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"
    BUDGET_RESET = "budget.reset"

    # Rate limit events
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"

    # Configuration events
    CONFIG_CHANGED = "config.changed"
    BACKEND_CHANGED = "backend.changed"
    PRICING_UPDATED = "pricing.updated"

    # Tracking events
    TRACKING_FAILURE = "tracking.failure"
    FALLBACK_ACTIVATED = "fallback.activated"

    # Cost events
    COST_RECORDED = "cost.recorded"
    COST_ANOMALY = "cost.anomaly"


@dataclass
class AuditEvent:
    """Represents an audit event."""

    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.now)
    actor: Optional[str] = None  # Who initiated the action (user/system)
    resource: Optional[str] = None  # What was affected (budget name, etc.)
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "resource": self.resource,
            "details": self.details,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditBackend(ABC):
    """Abstract base class for audit backends."""

    @abstractmethod
    def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        pass

    @abstractmethod
    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resource: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        pass


class LoggingAuditBackend(AuditBackend):
    """
    Audit backend that logs to Python logging.

    Suitable for development and when audit logs should go to
    existing log aggregation systems (CloudWatch, DataDog, etc.).
    """

    def __init__(
        self,
        logger_name: str = "llm_cost_guard.audit",
        log_level: int = logging.INFO,
    ):
        self._logger = logging.getLogger(logger_name)
        self._log_level = log_level
        self._events: list[AuditEvent] = []  # In-memory for query support
        self._max_events = 10000  # Limit in-memory storage

    def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        # Log to Python logger
        self._logger.log(
            self._log_level,
            f"AUDIT: {event.event_type.value} | resource={event.resource} | {event.details}"
        )

        # Store in memory for queries
        self._events.append(event)

        # Trim if too many events
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resource: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        results = []

        for event in reversed(self._events):  # Most recent first
            if event_type and event.event_type != event_type:
                continue
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue
            if resource and event.resource != resource:
                continue

            results.append(event)

            if len(results) >= limit:
                break

        return results


class FileAuditBackend(AuditBackend):
    """
    Audit backend that writes to a file (JSON Lines format).

    Suitable for compliance requirements where audit logs need
    to be stored in a specific location.

    Security:
    - Uses file locking for thread-safe concurrent writes
    - Atomic append operations
    """

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._lock = threading.Lock()

    def log(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Thread-safe: Uses locking for concurrent access.
        """
        import fcntl
        import os

        line = event.to_json() + "\n"

        with self._lock:
            try:
                # Open file for appending
                with open(self._file_path, "a") as f:
                    # Use file locking for process-level safety
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        f.write(line)
                        f.flush()
                        os.fsync(f.fileno())  # Ensure write is durable
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError as e:
                logger.error(f"Failed to write audit log: {e}")

    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resource: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events from file."""
        results = []

        try:
            with open(self._file_path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []

        for line in reversed(lines):  # Most recent first
            if not line.strip():
                continue

            data = json.loads(line)
            event = AuditEvent(
                event_type=AuditEventType(data["event_type"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                actor=data.get("actor"),
                resource=data.get("resource"),
                details=data.get("details", {}),
                metadata=data.get("metadata", {}),
            )

            if event_type and event.event_type != event_type:
                continue
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue
            if resource and event.resource != resource:
                continue

            results.append(event)

            if len(results) >= limit:
                break

        return results


class CompositeAuditBackend(AuditBackend):
    """
    Audit backend that writes to multiple backends.

    Useful for sending audit logs to both local storage and remote systems.
    """

    def __init__(self, backends: list[AuditBackend]):
        self._backends = backends

    def log(self, event: AuditEvent) -> None:
        """Log an audit event to all backends."""
        for backend in self._backends:
            try:
                backend.log(event)
            except Exception as e:
                logger.error(f"Audit backend failed: {e}")

    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resource: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query from first backend that succeeds."""
        for backend in self._backends:
            try:
                return backend.query(event_type, start_date, end_date, resource, limit)
            except Exception as e:
                logger.warning(f"Audit query failed on backend: {e}")
        return []


class AuditLogger:
    """
    Main audit logger for LLM Cost Guard.

    Usage:
        audit = AuditLogger(backend=FileAuditBackend("audit.log"))

        audit.log_budget_created(budget)
        audit.log_budget_exceeded(budget, current_spending)
    """

    def __init__(
        self,
        backend: Optional[AuditBackend] = None,
        enabled: bool = True,
        actor: Optional[str] = None,
    ):
        self._backend = backend or LoggingAuditBackend()
        self._enabled = enabled
        self._default_actor = actor or "system"
        self._event_callbacks: dict[AuditEventType, list[Callable[[AuditEvent], None]]] = {}

    def _log(self, event: AuditEvent) -> None:
        """Internal logging method."""
        if not self._enabled:
            return

        if event.actor is None:
            event.actor = self._default_actor

        self._backend.log(event)

        # Trigger callbacks
        callbacks = self._event_callbacks.get(event.event_type, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Audit callback failed: {e}")

    def on_event(
        self,
        event_type: AuditEventType,
        callback: Callable[[AuditEvent], None],
    ) -> None:
        """Register a callback for an event type."""
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        self._event_callbacks[event_type].append(callback)

    # Budget events
    def log_budget_created(
        self,
        budget_name: str,
        limit: float,
        period: str,
        action: str,
        actor: Optional[str] = None,
    ) -> None:
        """Log budget creation."""
        self._log(AuditEvent(
            event_type=AuditEventType.BUDGET_CREATED,
            actor=actor,
            resource=budget_name,
            details={
                "limit": limit,
                "period": period,
                "action": action,
            },
        ))

    def log_budget_modified(
        self,
        budget_name: str,
        old_values: dict[str, Any],
        new_values: dict[str, Any],
        actor: Optional[str] = None,
    ) -> None:
        """Log budget modification."""
        self._log(AuditEvent(
            event_type=AuditEventType.BUDGET_MODIFIED,
            actor=actor,
            resource=budget_name,
            details={
                "old_values": old_values,
                "new_values": new_values,
            },
        ))

    def log_budget_deleted(
        self,
        budget_name: str,
        actor: Optional[str] = None,
    ) -> None:
        """Log budget deletion."""
        self._log(AuditEvent(
            event_type=AuditEventType.BUDGET_DELETED,
            actor=actor,
            resource=budget_name,
        ))

    def log_budget_warning(
        self,
        budget_name: str,
        current_spending: float,
        limit: float,
        utilization: float,
    ) -> None:
        """Log budget warning."""
        self._log(AuditEvent(
            event_type=AuditEventType.BUDGET_WARNING,
            resource=budget_name,
            details={
                "current_spending": current_spending,
                "limit": limit,
                "utilization_percent": utilization * 100,
            },
        ))

    def log_budget_exceeded(
        self,
        budget_name: str,
        current_spending: float,
        limit: float,
        action_taken: str,
    ) -> None:
        """Log budget exceeded event."""
        self._log(AuditEvent(
            event_type=AuditEventType.BUDGET_EXCEEDED,
            resource=budget_name,
            details={
                "current_spending": current_spending,
                "limit": limit,
                "action_taken": action_taken,
            },
        ))

    def log_budget_reset(
        self,
        budget_name: str,
        reason: str = "period_end",
        actor: Optional[str] = None,
    ) -> None:
        """Log budget reset."""
        self._log(AuditEvent(
            event_type=AuditEventType.BUDGET_RESET,
            actor=actor,
            resource=budget_name,
            details={"reason": reason},
        ))

    # Rate limit events
    def log_rate_limit_exceeded(
        self,
        limit_name: str,
        current: int,
        limit: int,
        retry_after: float,
    ) -> None:
        """Log rate limit exceeded event."""
        self._log(AuditEvent(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            resource=limit_name,
            details={
                "current": current,
                "limit": limit,
                "retry_after_seconds": retry_after,
            },
        ))

    # Tracking events
    def log_tracking_failure(
        self,
        error: str,
        backend: str,
        action_taken: str,
    ) -> None:
        """Log tracking failure."""
        self._log(AuditEvent(
            event_type=AuditEventType.TRACKING_FAILURE,
            resource=backend,
            details={
                "error": error,
                "action_taken": action_taken,
            },
        ))

    def log_fallback_activated(
        self,
        original_backend: str,
        fallback_backend: str,
        reason: str,
    ) -> None:
        """Log fallback activation."""
        self._log(AuditEvent(
            event_type=AuditEventType.FALLBACK_ACTIVATED,
            details={
                "original_backend": original_backend,
                "fallback_backend": fallback_backend,
                "reason": reason,
            },
        ))

    # Query methods
    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resource: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        return self._backend.query(event_type, start_date, end_date, resource, limit)

    def get_budget_history(
        self,
        budget_name: str,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Get audit history for a specific budget."""
        return self._backend.query(resource=budget_name, limit=limit)
