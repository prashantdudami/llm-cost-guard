"""
End-to-end tests for LLM Cost Guard.

These tests simulate complete workflows without making actual API calls.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from llm_cost_guard import CostTracker, Budget, BudgetAction, RateLimit
from llm_cost_guard.exceptions import BudgetExceededError, RateLimitExceededError


class TestCompleteWorkflow:
    """End-to-end workflow tests."""

    def test_basic_tracking_workflow(self):
        """Test complete workflow: track -> query -> report."""
        tracker = CostTracker(backend="memory")

        # Simulate multiple API calls
        for i in range(10):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                tags={"user": f"user_{i % 3}"},
            )

        # Verify tracking
        report = tracker.daily_report()
        assert report.total_calls == 10
        assert report.total_cost > 0

        # Query by tag
        user_0_report = tracker.get_costs(tags={"user": "user_0"})
        assert user_0_report.total_calls >= 3

        tracker.close()

    def test_multi_provider_workflow(self):
        """Test tracking across multiple providers."""
        tracker = CostTracker(backend="memory")

        # OpenAI calls
        for _ in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )

        # Anthropic calls
        for _ in range(5):
            tracker.record(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                input_tokens=100,
                output_tokens=50,
            )

        # Bedrock calls
        for _ in range(5):
            tracker.record(
                provider="bedrock",
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                input_tokens=100,
                output_tokens=50,
            )

        report = tracker.daily_report()
        assert report.total_calls == 15

        # Group by provider
        report = tracker.get_costs(group_by=["provider"])
        groups = report.grouped_data.get("groups", [])
        assert len(groups) == 3  # Three providers

        tracker.close()

    def test_budget_enforcement_workflow(self):
        """Test complete budget enforcement workflow."""
        exceeded_events = []
        warning_events = []

        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(
                    name="session",
                    limit=1.00,
                    period="day",
                    action=BudgetAction.BLOCK,
                    warning_threshold=0.5,
                ),
            ],
        )

        @tracker.on_budget_warning
        def on_warning(budget, current):
            warning_events.append((budget.name, current))

        @tracker.on_budget_exceeded
        def on_exceeded(budget):
            exceeded_events.append(budget.name)

        # Track until we exceed budget
        try:
            for i in range(100):
                tracker.record(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=10000,  # High tokens per call
                    output_tokens=5000,
                )
        except BudgetExceededError:
            pass  # Expected

        # Verify warning was triggered before block
        assert len(warning_events) >= 1

        tracker.close()

    def test_rate_limiting_workflow(self):
        """Test rate limiting workflow."""
        tracker = CostTracker(
            backend="memory",
            rate_limits=[
                RateLimit(
                    name="calls-per-minute",
                    limit=5,
                    period="minute",
                    scope="global",
                ),
            ],
        )

        # Make calls up to limit
        for i in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )

        # Next call should be rate limited
        with pytest.raises(RateLimitExceededError):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )

        tracker.close()

    def test_span_tracking_workflow(self):
        """Test hierarchical span tracking for agents."""
        tracker = CostTracker(backend="memory")

        # Simulate an agent workflow with multiple tool calls
        with tracker.span("agent_workflow", tags={"task": "research"}) as agent_span:
            # Retrieval step
            with tracker.span("retrieval") as retrieval_span:
                tracker.record(
                    provider="openai",
                    model="text-embedding-3-small",
                    input_tokens=500,
                    output_tokens=0,
                )

            # Reasoning steps
            with tracker.span("reasoning") as reasoning_span:
                for _ in range(3):
                    tracker.record(
                        provider="openai",
                        model="gpt-4o",
                        input_tokens=2000,
                        output_tokens=500,
                    )

            # Final response
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=3000,
                output_tokens=1000,
            )

        # Verify span tracking
        assert retrieval_span.call_count == 1
        assert reasoning_span.call_count == 3
        assert agent_span.call_count == 5  # All calls in the workflow
        assert len(agent_span.children) == 2  # retrieval and reasoning spans

        tracker.close()

    def test_attribution_workflow(self):
        """Test cost attribution with tags."""
        tracker = CostTracker(backend="memory")

        teams = ["search", "chat", "analytics"]
        features = ["autocomplete", "summarize", "translate"]

        # Simulate traffic with various attributions
        for i in range(30):
            team = teams[i % 3]
            feature = features[i % 3]
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100 * (i + 1),
                output_tokens=50 * (i + 1),
                tags={"team": team, "feature": feature},
            )

        # Query by team
        for team in teams:
            report = tracker.get_costs(tags={"team": team})
            assert report.total_calls == 10

        # Group by team and feature
        report = tracker.get_costs(group_by=["team", "feature"])
        groups = report.grouped_data.get("groups", [])
        assert len(groups) > 0

        tracker.close()

    def test_sqlite_persistence_workflow(self):
        """Test SQLite backend for persistence."""
        import tempfile
        import os

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # First session: write data
            tracker1 = CostTracker(backend=f"sqlite:///{db_path}")

            for i in range(5):
                tracker1.record(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                )

            tracker1.close()

            # Second session: read data
            tracker2 = CostTracker(backend=f"sqlite:///{db_path}")

            report = tracker2.daily_report()
            assert report.total_calls == 5

            tracker2.close()

        finally:
            os.unlink(db_path)

    def test_failed_calls_tracking(self):
        """Test tracking of failed API calls."""
        tracker = CostTracker(backend="memory", track_failed_calls=True)

        # Successful calls
        for _ in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                success=True,
            )

        # Failed calls
        for _ in range(3):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=0,  # No output on failure
                success=False,
                error_type="RateLimitError",
            )

        report = tracker.daily_report()
        assert report.total_calls == 8
        assert report.successful_calls == 5
        assert report.failed_calls == 3

        tracker.close()

    def test_cache_tracking_workflow(self):
        """Test tracking of cached responses."""
        tracker = CostTracker(backend="memory", track_cache_savings=True)

        # Non-cached calls
        for _ in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
                cached_tokens=0,
            )

        # Cached calls
        for _ in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
                cached_tokens=500,  # Half the input was cached
            )

        report = tracker.daily_report()
        assert report.total_calls == 10

        # Cached calls should have lower cost
        records = tracker._backend.get_records()
        cached_records = [r for r in records if r.cached]
        non_cached_records = [r for r in records if not r.cached]

        assert len(cached_records) == 5
        assert len(non_cached_records) == 5

        tracker.close()


class TestDecoratorE2E:
    """End-to-end tests for decorator-based tracking."""

    def test_sync_decorator_e2e(self):
        """Test sync function decorator end-to-end."""
        tracker = CostTracker(backend="memory")

        @tracker.track(tags={"feature": "test"})
        def mock_llm_call():
            response = MagicMock()
            response.model = "gpt-4o"
            response.usage = MagicMock()
            response.usage.prompt_tokens = 150
            response.usage.completion_tokens = 75
            response.usage.total_tokens = 225
            response.usage.prompt_tokens_details = None
            return response

        # Make several calls
        for _ in range(3):
            mock_llm_call()

        report = tracker.daily_report()
        assert report.total_calls == 3

        tracker.close()

    @pytest.mark.asyncio
    async def test_async_decorator_e2e(self):
        """Test async function decorator end-to-end."""
        tracker = CostTracker(backend="memory")

        @tracker.track(tags={"feature": "async_test"})
        async def mock_async_llm_call():
            response = MagicMock()
            response.model = "gpt-4o"
            response.usage = MagicMock()
            response.usage.prompt_tokens = 150
            response.usage.completion_tokens = 75
            response.usage.total_tokens = 225
            response.usage.prompt_tokens_details = None
            return response

        # Make several async calls
        for _ in range(3):
            await mock_async_llm_call()

        report = tracker.daily_report()
        assert report.total_calls == 3

        tracker.close()


class TestHealthAndReporting:
    """End-to-end tests for health checks and reporting."""

    def test_health_check_workflow(self):
        """Test health check returns accurate status."""
        tracker = CostTracker(backend="memory")

        # Initially healthy
        health = tracker.health_check()
        assert health.healthy is True
        assert health.backend_connected is True
        assert health.pricing_fresh is True

        # Make some calls
        for _ in range(3):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )

        # Check health after activity
        health = tracker.health_check()
        assert health.healthy is True
        assert health.last_record_time is not None

        tracker.close()

    def test_report_generation_workflow(self):
        """Test report generation with various groupings."""
        tracker = CostTracker(backend="memory")

        # Create diverse data
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        for i, model in enumerate(models):
            for j in range(5):
                tracker.record(
                    provider="openai",
                    model=model,
                    input_tokens=100 * (i + 1),
                    output_tokens=50 * (i + 1),
                    tags={"priority": "high" if j % 2 == 0 else "low"},
                )

        # Daily report
        daily = tracker.daily_report()
        assert daily.total_calls == 15

        # By model
        by_model = tracker.report_by_model(period="day")
        assert by_model.total_calls == 15

        # Trend analysis
        trends = tracker.trend_analysis(metric="cost", granularity="hour", last_n_days=1)
        assert "data" in trends

        tracker.close()


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_zero_tokens(self):
        """Test handling of zero token calls."""
        tracker = CostTracker(backend="memory")

        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
        )

        assert record.total_cost == 0.0

        tracker.close()

    def test_very_large_token_counts(self):
        """Test handling of very large token counts."""
        tracker = CostTracker(backend="memory")

        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )

        assert record.total_cost > 0
        assert record.input_tokens == 1_000_000

        tracker.close()

    def test_concurrent_tracking(self):
        """Test thread-safe concurrent tracking."""
        import threading

        tracker = CostTracker(backend="memory")
        errors = []

        def make_calls(thread_id):
            try:
                for i in range(10):
                    tracker.record(
                        provider="openai",
                        model="gpt-4o",
                        input_tokens=100,
                        output_tokens=50,
                        tags={"thread": str(thread_id)},
                    )
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=make_calls, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0

        # All calls should be recorded
        report = tracker.daily_report()
        assert report.total_calls == 50  # 5 threads * 10 calls each

        tracker.close()

    def test_tracking_failure_allow_mode(self):
        """Test that tracking failures are handled gracefully in allow mode."""
        tracker = CostTracker(
            backend="memory",
            on_tracking_failure="allow",
        )

        # Normal operation should work
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        assert tracker.last_call() is not None

        tracker.close()


class TestAuditWorkflow:
    """End-to-end tests for audit logging."""

    def test_complete_audit_trail(self):
        """Test complete audit trail for budget lifecycle."""
        from llm_cost_guard.audit import LoggingAuditBackend, AuditEventType
        
        audit_backend = LoggingAuditBackend()
        
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(
                    name="daily",
                    limit=1.00,
                    action=BudgetAction.BLOCK,
                    warning_threshold=0.5,
                ),
            ],
            audit_backend=audit_backend,
        )
        
        # Should have logged budget creation
        creation_events = audit_backend.query(event_type=AuditEventType.BUDGET_CREATED)
        assert len(creation_events) == 1
        assert creation_events[0].resource == "daily"
        
        # Make some calls until we exceed budget
        try:
            for i in range(50):
                tracker.record(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=10000,
                    output_tokens=5000,
                )
        except BudgetExceededError:
            pass  # Expected
        
        # Should have warning and exceeded events
        warning_events = audit_backend.query(event_type=AuditEventType.BUDGET_WARNING)
        exceeded_events = audit_backend.query(event_type=AuditEventType.BUDGET_EXCEEDED)
        
        assert len(exceeded_events) >= 1
        
        tracker.close()

    def test_audit_query_by_resource(self):
        """Test querying audit by resource."""
        from llm_cost_guard.audit import LoggingAuditBackend
        
        audit_backend = LoggingAuditBackend()
        
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(name="daily", limit=100.0, action=BudgetAction.WARN),
                Budget(name="monthly", limit=1000.0, action=BudgetAction.WARN),
            ],
            audit_backend=audit_backend,
        )
        
        # Get history for specific budget
        daily_history = tracker.audit.get_budget_history("daily")
        monthly_history = tracker.audit.get_budget_history("monthly")
        
        assert len(daily_history) == 1  # Creation event
        assert len(monthly_history) == 1  # Creation event
        
        tracker.close()


class TestMetricsWorkflow:
    """End-to-end tests for metrics and observability."""

    def test_metrics_accumulation(self):
        """Test metrics accumulate correctly over workflow."""
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(name="session", limit=10.0, action=BudgetAction.BLOCK),
            ],
        )
        
        # Track some calls
        for _ in range(10):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        
        metrics = tracker.get_metrics()
        
        assert metrics["budget_checks"] == 10
        assert metrics["using_fallback"] is False
        assert metrics["backend_failures"] == 0
        
        tracker.close()

    def test_health_check_with_activity(self):
        """Test health check reflects activity."""
        tracker = CostTracker(backend="memory")
        
        # Initially healthy
        health = tracker.health_check()
        assert health.healthy is True
        assert health.last_record_time is None
        
        # Make a call
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        
        # Health should reflect activity
        health = tracker.health_check()
        assert health.healthy is True
        assert health.last_record_time is not None
        
        tracker.close()

    def test_metrics_with_errors(self):
        """Test metrics track errors correctly."""
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(name="tiny", limit=0.001, action=BudgetAction.BLOCK),
            ],
        )
        
        # First call might succeed
        try:
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        except BudgetExceededError:
            pass
        
        # Second call should fail
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


class TestGracefulDegradation:
    """End-to-end tests for graceful degradation."""

    def test_fallback_mode_workflow(self):
        """Test complete workflow in fallback mode."""
        from unittest.mock import patch
        
        with patch("llm_cost_guard.backends.get_backend") as mock_get_backend:
            mock_get_backend.side_effect = Exception("Backend unavailable")
            
            tracker = CostTracker(
                backend="redis://localhost:6379",
                on_tracking_failure="fallback",
            )
            
            # Should still work with fallback
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
            
            # Health should report fallback
            health = tracker.health_check()
            assert health.healthy is False
            
            # Metrics should show fallback
            metrics = tracker.get_metrics()
            assert metrics["using_fallback"] is True
            assert metrics["fallback_activations"] == 1
            
            tracker.close()

    def test_allow_mode_continues(self):
        """Test allow mode continues on errors."""
        from unittest.mock import patch
        
        with patch("llm_cost_guard.backends.get_backend") as mock_get_backend:
            mock_get_backend.side_effect = Exception("Backend unavailable")
            
            tracker = CostTracker(
                backend="redis://localhost:6379",
                on_tracking_failure="allow",
            )
            
            # Should still work
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
            
            assert tracker.last_call() is not None
            
            tracker.close()
