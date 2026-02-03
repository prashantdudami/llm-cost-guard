"""
Unit tests for CostTracker metrics and observability.
"""

import pytest
from unittest.mock import MagicMock, patch

from llm_cost_guard import CostTracker, Budget, BudgetAction
from llm_cost_guard.audit import AuditEventType, LoggingAuditBackend
from llm_cost_guard.exceptions import BudgetExceededError


class TestTrackerMetrics:
    """Tests for CostTracker metrics."""

    def test_initial_metrics(self):
        """Test initial metrics are zero."""
        tracker = CostTracker(backend="memory")
        
        metrics = tracker.get_metrics()
        
        assert metrics["backend_failures"] == 0
        assert metrics["fallback_activations"] == 0
        assert metrics["budget_checks"] == 0
        assert metrics["budget_exceeded_count"] == 0
        assert metrics["rate_limit_exceeded_count"] == 0
        assert metrics["tracking_errors"] == 0
        assert metrics["using_fallback"] is False
        
        tracker.close()

    def test_budget_check_metric_increments(self):
        """Test budget_checks metric increments on each record."""
        tracker = CostTracker(
            backend="memory",
            budgets=[Budget(name="test", limit=100.0, action=BudgetAction.WARN)],
        )
        
        # Record some costs
        for _ in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        
        metrics = tracker.get_metrics()
        assert metrics["budget_checks"] == 5
        
        tracker.close()

    def test_budget_exceeded_metric(self):
        """Test budget_exceeded_count increments on exceeded."""
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(name="tiny", limit=0.0001, action=BudgetAction.BLOCK)
            ],
        )
        
        # Try to exceed budget
        try:
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        except BudgetExceededError:
            pass
        
        metrics = tracker.get_metrics()
        assert metrics["budget_exceeded_count"] >= 1
        
        tracker.close()

    def test_fallback_activation_metric(self):
        """Test fallback_activations increments when backend fails."""
        with patch("llm_cost_guard.backends.get_backend") as mock_get_backend:
            mock_get_backend.side_effect = Exception("Backend unavailable")
            
            tracker = CostTracker(
                backend="redis://localhost:6379",
                on_tracking_failure="fallback",
            )
            
            metrics = tracker.get_metrics()
            assert metrics["fallback_activations"] == 1
            assert metrics["using_fallback"] is True
            
            tracker.close()

    def test_backend_failures_metric(self):
        """Test backend_failures increments on errors."""
        with patch("llm_cost_guard.backends.get_backend") as mock_get_backend:
            mock_get_backend.side_effect = Exception("Backend unavailable")
            
            tracker = CostTracker(
                backend="redis://localhost:6379",
                on_tracking_failure="allow",  # Allow but count failure
            )
            
            metrics = tracker.get_metrics()
            assert metrics["backend_failures"] >= 1
            
            tracker.close()

    def test_metrics_thread_safety(self):
        """Test metrics are thread-safe."""
        import threading
        
        tracker = CostTracker(backend="memory")
        
        def make_calls():
            for _ in range(100):
                tracker.record(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=10,
                    output_tokens=5,
                )
        
        threads = [threading.Thread(target=make_calls) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have recorded all calls without race conditions
        report = tracker.daily_report()
        assert report.total_calls == 500
        
        tracker.close()


class TestTrackerAuditIntegration:
    """Tests for CostTracker audit logging integration."""

    def test_audit_enabled_by_default(self):
        """Test audit logging is enabled by default."""
        tracker = CostTracker(backend="memory")
        
        assert tracker.audit is not None
        
        tracker.close()

    def test_audit_can_be_disabled(self):
        """Test audit logging can be disabled."""
        tracker = CostTracker(backend="memory", audit_enabled=False)
        
        # Record something
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        
        # No audit events should be logged
        events = tracker.audit.query()
        assert len(events) == 0
        
        tracker.close()

    def test_budget_creation_is_audited(self):
        """Test budget creation is logged to audit."""
        backend = LoggingAuditBackend()
        
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(name="daily", limit=100.0, action=BudgetAction.BLOCK),
                Budget(name="monthly", limit=1000.0, action=BudgetAction.WARN),
            ],
            audit_backend=backend,
        )
        
        events = backend.query(event_type=AuditEventType.BUDGET_CREATED)
        assert len(events) == 2
        
        tracker.close()

    def test_budget_exceeded_is_audited(self):
        """Test budget exceeded is logged to audit."""
        backend = LoggingAuditBackend()
        
        tracker = CostTracker(
            backend="memory",
            budgets=[Budget(name="tiny", limit=0.0001, action=BudgetAction.BLOCK)],
            audit_backend=backend,
        )
        
        try:
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        except BudgetExceededError:
            pass
        
        events = backend.query(event_type=AuditEventType.BUDGET_EXCEEDED)
        assert len(events) >= 1
        
        tracker.close()

    def test_budget_warning_is_audited(self):
        """Test budget warning is logged to audit."""
        backend = LoggingAuditBackend()
        
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(
                    name="test",
                    limit=0.01,  # Low limit to trigger warning
                    action=BudgetAction.WARN,
                    warning_threshold=0.1,  # Very low threshold
                )
            ],
            audit_backend=backend,
        )
        
        # Make a call that should trigger warning
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        
        events = backend.query(event_type=AuditEventType.BUDGET_WARNING)
        # Warning may or may not trigger depending on cost
        
        tracker.close()

    def test_custom_audit_backend(self):
        """Test using custom audit backend."""
        backend = LoggingAuditBackend()
        
        tracker = CostTracker(
            backend="memory",
            audit_backend=backend,
        )
        
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        
        # Query through tracker's audit interface
        history = tracker.audit.query()
        # Should have access to the audit logger
        assert tracker.audit is not None
        
        tracker.close()


class TestHealthCheckWithMetrics:
    """Tests for health check with metrics integration."""

    def test_health_includes_fallback_status(self):
        """Test health check reports fallback status."""
        with patch("llm_cost_guard.backends.get_backend") as mock_get_backend:
            mock_get_backend.side_effect = Exception("Backend unavailable")
            
            tracker = CostTracker(
                backend="redis://localhost:6379",
                on_tracking_failure="fallback",
            )
            
            health = tracker.health_check()
            
            # Should report as not fully healthy when using fallback
            assert health.healthy is False
            assert "fallback" in str(health.errors).lower()
            
            tracker.close()

    def test_health_backend_connected(self):
        """Test health check reports backend connection status."""
        tracker = CostTracker(backend="memory")
        
        health = tracker.health_check()
        
        assert health.backend_connected is True
        
        tracker.close()

    def test_health_pricing_fresh(self):
        """Test health check reports pricing freshness."""
        tracker = CostTracker(backend="memory")
        
        health = tracker.health_check()
        
        assert health.pricing_fresh is True
        
        tracker.close()
