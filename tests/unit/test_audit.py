"""
Unit tests for audit logging module.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from llm_cost_guard.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    CompositeAuditBackend,
    FileAuditBackend,
    LoggingAuditBackend,
)


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_create_event(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_type=AuditEventType.BUDGET_CREATED,
            actor="admin",
            resource="daily-budget",
            details={"limit": 100.0, "period": "day"},
        )

        assert event.event_type == AuditEventType.BUDGET_CREATED
        assert event.actor == "admin"
        assert event.resource == "daily-budget"
        assert event.details["limit"] == 100.0

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = AuditEvent(
            event_type=AuditEventType.BUDGET_EXCEEDED,
            resource="monthly",
            details={"current": 150.0, "limit": 100.0},
        )

        data = event.to_dict()

        assert data["event_type"] == "budget.exceeded"
        assert data["resource"] == "monthly"
        assert "timestamp" in data

    def test_event_to_json(self):
        """Test converting event to JSON."""
        event = AuditEvent(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            resource="api-limit",
            details={"current": 100, "limit": 50},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["event_type"] == "rate_limit.exceeded"
        assert parsed["resource"] == "api-limit"


class TestLoggingAuditBackend:
    """Tests for LoggingAuditBackend."""

    def test_log_event(self):
        """Test logging an event."""
        backend = LoggingAuditBackend()

        event = AuditEvent(
            event_type=AuditEventType.BUDGET_CREATED,
            resource="test-budget",
            details={"limit": 50.0},
        )

        backend.log(event)

        # Verify event was stored
        events = backend.query()
        assert len(events) == 1
        assert events[0].resource == "test-budget"

    def test_query_by_event_type(self):
        """Test querying by event type."""
        backend = LoggingAuditBackend()

        # Log different event types
        backend.log(AuditEvent(
            event_type=AuditEventType.BUDGET_CREATED,
            resource="budget1",
        ))
        backend.log(AuditEvent(
            event_type=AuditEventType.BUDGET_EXCEEDED,
            resource="budget1",
        ))
        backend.log(AuditEvent(
            event_type=AuditEventType.BUDGET_CREATED,
            resource="budget2",
        ))

        # Query only BUDGET_CREATED
        events = backend.query(event_type=AuditEventType.BUDGET_CREATED)
        assert len(events) == 2

    def test_query_by_resource(self):
        """Test querying by resource."""
        backend = LoggingAuditBackend()

        backend.log(AuditEvent(
            event_type=AuditEventType.BUDGET_WARNING,
            resource="daily",
        ))
        backend.log(AuditEvent(
            event_type=AuditEventType.BUDGET_WARNING,
            resource="monthly",
        ))
        backend.log(AuditEvent(
            event_type=AuditEventType.BUDGET_EXCEEDED,
            resource="daily",
        ))

        events = backend.query(resource="daily")
        assert len(events) == 2

    def test_query_with_limit(self):
        """Test query with limit."""
        backend = LoggingAuditBackend()

        for i in range(10):
            backend.log(AuditEvent(
                event_type=AuditEventType.COST_RECORDED,
                resource=f"record-{i}",
            ))

        events = backend.query(limit=5)
        assert len(events) == 5

    def test_max_events_limit(self):
        """Test that old events are evicted when limit reached."""
        backend = LoggingAuditBackend()
        backend._max_events = 10  # Set low limit for testing

        for i in range(15):
            backend.log(AuditEvent(
                event_type=AuditEventType.COST_RECORDED,
                resource=f"record-{i}",
            ))

        # Should have trimmed to max
        assert len(backend._events) <= 10


class TestFileAuditBackend:
    """Tests for FileAuditBackend."""

    def test_log_to_file(self):
        """Test logging to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            path = f.name

        try:
            backend = FileAuditBackend(path)

            event = AuditEvent(
                event_type=AuditEventType.BUDGET_CREATED,
                actor="test",
                resource="test-budget",
                details={"limit": 100.0},
            )

            backend.log(event)

            # Verify file content
            with open(path, "r") as f:
                content = f.read()
                assert "budget.created" in content
                assert "test-budget" in content

        finally:
            os.unlink(path)

    def test_query_from_file(self):
        """Test querying from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            path = f.name

        try:
            backend = FileAuditBackend(path)

            # Log some events
            for i in range(5):
                backend.log(AuditEvent(
                    event_type=AuditEventType.BUDGET_WARNING,
                    resource=f"budget-{i}",
                ))

            # Query back
            events = backend.query()
            assert len(events) == 5

        finally:
            os.unlink(path)

    def test_query_nonexistent_file(self):
        """Test querying from non-existent file."""
        backend = FileAuditBackend("/nonexistent/path/audit.log")

        events = backend.query()
        assert events == []


class TestCompositeAuditBackend:
    """Tests for CompositeAuditBackend."""

    def test_log_to_multiple_backends(self):
        """Test logging to multiple backends."""
        backend1 = LoggingAuditBackend()
        backend2 = LoggingAuditBackend()

        composite = CompositeAuditBackend([backend1, backend2])

        event = AuditEvent(
            event_type=AuditEventType.BUDGET_EXCEEDED,
            resource="test",
        )

        composite.log(event)

        # Both backends should have the event
        assert len(backend1.query()) == 1
        assert len(backend2.query()) == 1

    def test_query_from_first_backend(self):
        """Test query returns from first working backend."""
        backend1 = LoggingAuditBackend()
        backend2 = LoggingAuditBackend()

        # Add events to backend1
        backend1.log(AuditEvent(
            event_type=AuditEventType.BUDGET_CREATED,
            resource="test",
        ))

        composite = CompositeAuditBackend([backend1, backend2])

        events = composite.query()
        assert len(events) == 1


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_log_budget_created(self):
        """Test logging budget creation."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend)

        logger.log_budget_created(
            budget_name="daily",
            limit=100.0,
            period="day",
            action="block",
            actor="admin",
        )

        events = backend.query()
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.BUDGET_CREATED
        assert events[0].resource == "daily"

    def test_log_budget_exceeded(self):
        """Test logging budget exceeded."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend)

        logger.log_budget_exceeded(
            budget_name="monthly",
            current_spending=150.0,
            limit=100.0,
            action_taken="blocked",
        )

        events = backend.query()
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.BUDGET_EXCEEDED
        assert events[0].details["current_spending"] == 150.0

    def test_log_budget_warning(self):
        """Test logging budget warning."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend)

        logger.log_budget_warning(
            budget_name="daily",
            current_spending=80.0,
            limit=100.0,
            utilization=0.8,
        )

        events = backend.query()
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.BUDGET_WARNING
        assert events[0].details["utilization_percent"] == 80.0

    def test_log_rate_limit_exceeded(self):
        """Test logging rate limit exceeded."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend)

        logger.log_rate_limit_exceeded(
            limit_name="api-calls",
            current=100,
            limit=50,
            retry_after=30.0,
        )

        events = backend.query()
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.RATE_LIMIT_EXCEEDED
        assert events[0].details["retry_after_seconds"] == 30.0

    def test_log_tracking_failure(self):
        """Test logging tracking failure."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend)

        logger.log_tracking_failure(
            error="Connection refused",
            backend="redis://localhost",
            action_taken="allowed",
        )

        events = backend.query()
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.TRACKING_FAILURE

    def test_log_fallback_activated(self):
        """Test logging fallback activation."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend)

        logger.log_fallback_activated(
            original_backend="redis://localhost",
            fallback_backend="memory",
            reason="Connection refused",
        )

        events = backend.query()
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.FALLBACK_ACTIVATED

    def test_disabled_logging(self):
        """Test that disabled logger doesn't log."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend, enabled=False)

        logger.log_budget_created(
            budget_name="test",
            limit=100.0,
            period="day",
            action="block",
        )

        events = backend.query()
        assert len(events) == 0

    def test_event_callback(self):
        """Test event callback registration."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend)

        callback_events = []

        def on_exceeded(event):
            callback_events.append(event)

        logger.on_event(AuditEventType.BUDGET_EXCEEDED, on_exceeded)

        logger.log_budget_exceeded(
            budget_name="test",
            current_spending=150.0,
            limit=100.0,
            action_taken="blocked",
        )

        assert len(callback_events) == 1
        assert callback_events[0].event_type == AuditEventType.BUDGET_EXCEEDED

    def test_get_budget_history(self):
        """Test getting budget history."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend)

        # Log events for different budgets
        logger.log_budget_created("daily", 100.0, "day", "block")
        logger.log_budget_warning("daily", 80.0, 100.0, 0.8)
        logger.log_budget_exceeded("daily", 110.0, 100.0, "blocked")
        logger.log_budget_created("monthly", 1000.0, "month", "warn")

        # Get history for daily budget
        history = logger.get_budget_history("daily")
        assert len(history) == 3

    def test_default_actor(self):
        """Test default actor is set."""
        backend = LoggingAuditBackend()
        logger = AuditLogger(backend=backend, actor="system-service")

        logger.log_budget_created("test", 100.0, "day", "block")

        events = backend.query()
        assert events[0].actor == "system-service"
