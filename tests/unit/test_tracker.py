"""
Unit tests for CostTracker.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from llm_cost_guard import CostTracker, Budget, BudgetAction
from llm_cost_guard.exceptions import BudgetExceededError, RateLimitExceededError
from llm_cost_guard.models import CostRecord


class TestCostTrackerBasic:
    """Basic CostTracker functionality tests."""

    def test_init_default(self):
        """Test default initialization."""
        tracker = CostTracker()
        assert tracker is not None
        tracker.close()

    def test_init_with_memory_backend(self):
        """Test initialization with memory backend."""
        tracker = CostTracker(backend="memory")
        assert tracker is not None
        tracker.close()

    def test_record_basic(self, tracker):
        """Test basic cost recording."""
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        assert record is not None
        assert record.provider == "openai"
        assert record.model == "gpt-4o"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.total_cost > 0

    def test_record_with_tags(self, tracker):
        """Test recording with tags."""
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            tags={"team": "search", "feature": "autocomplete"},
        )

        assert record.tags["team"] == "search"
        assert record.tags["feature"] == "autocomplete"

    def test_last_call(self, tracker):
        """Test last_call() returns the most recent record."""
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        last = tracker.last_call()
        assert last is not None
        assert last.model == "gpt-4o"

    def test_record_rejects_invalid_provider(self):
        """Test that invalid provider names are rejected."""
        tracker = CostTracker()
        
        with pytest.raises(ValueError, match="provider"):
            tracker.record(
                provider="",  # Empty
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        
        tracker.close()

    def test_record_rejects_negative_tokens(self):
        """Test that negative token counts are rejected."""
        tracker = CostTracker()
        
        with pytest.raises(ValueError, match="input_tokens must be >= 0"):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=-1,
                output_tokens=50,
            )
        
        tracker.close()

    def test_get_costs(self, tracker):
        """Test get_costs() returns a report."""
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        report = tracker.get_costs()
        assert report is not None
        assert report.total_calls >= 1
        assert report.total_cost > 0


class TestCostTrackerDecorator:
    """Tests for the @tracker.track decorator."""

    def test_decorator_sync_function(self, tracker):
        """Test decorator with sync function."""

        @tracker.track
        def mock_llm_call():
            # Return a mock response
            mock_response = MagicMock()
            mock_response.model = "gpt-4o"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.usage.prompt_tokens_details = None
            return mock_response

        result = mock_llm_call()
        assert result is not None

    def test_decorator_with_tags(self, tracker):
        """Test decorator with tags."""

        @tracker.track(tags={"team": "search"})
        def mock_llm_call():
            mock_response = MagicMock()
            mock_response.model = "gpt-4o"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.usage.prompt_tokens_details = None
            return mock_response

        result = mock_llm_call()
        assert result is not None

    @pytest.mark.asyncio
    async def test_decorator_async_function(self, tracker):
        """Test decorator with async function."""

        @tracker.track
        async def async_mock_llm_call():
            mock_response = MagicMock()
            mock_response.model = "gpt-4o"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.usage.prompt_tokens_details = None
            return mock_response

        result = await async_mock_llm_call()
        assert result is not None


class TestCostTrackerBudget:
    """Tests for budget enforcement."""

    def test_budget_warning(self):
        """Test budget warning callback."""
        warnings = []

        # Create tracker with only daily budget (no per-request limit)
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(
                    name="daily",
                    limit=10.00,
                    period="day",
                    action=BudgetAction.WARN,
                    warning_threshold=0.5,  # Lower threshold for easier testing
                ),
            ],
        )

        @tracker.on_budget_warning
        def handle_warning(budget, current):
            warnings.append((budget.name, current))

        # Record a cost that should trigger warning (50%+ of $10)
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=200000,  # High token count
            output_tokens=100000,
        )

        # Should trigger warning since we're over 50% of budget

    def test_budget_exceeded_block(self):
        """Test budget exceeded with BLOCK action."""
        tracker = CostTracker(
            backend="memory",
            budgets=[
                Budget(
                    name="tiny-budget",
                    limit=0.001,  # Very small
                    period="day",
                    action=BudgetAction.BLOCK,
                )
            ],
        )

        # First call might work
        try:
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        except BudgetExceededError:
            pass  # Expected

        # Second call should definitely fail
        with pytest.raises(BudgetExceededError):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )

    def test_budget_utilization(self, tracker_with_budget):
        """Test budget utilization tracking."""
        tracker_with_budget.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        utilization = tracker_with_budget.get_budget_utilization("daily")
        assert utilization >= 0


class TestCostTrackerHealth:
    """Tests for health check."""

    def test_health_check(self, tracker):
        """Test health check returns valid status."""
        health = tracker.health_check()

        assert health is not None
        assert health.backend_connected is True
        assert isinstance(health.pricing_fresh, bool)


class TestCostTrackerSpan:
    """Tests for hierarchical span tracking."""

    def test_span_basic(self, tracker):
        """Test basic span creation and usage."""
        with tracker.span("test-operation") as span:
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=200,
                output_tokens=100,
            )

        assert span.call_count == 2
        assert span.total_cost > 0
        assert "gpt-4o" in span.models_used

    def test_nested_spans(self, tracker):
        """Test nested span tracking."""
        with tracker.span("outer") as outer:
            with tracker.span("inner") as inner:
                tracker.record(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                )

            assert inner.call_count == 1

        assert outer.call_count == 1
        assert len(outer.children) == 1
