"""
Edge case tests for LLM Cost Guard.

Tests for boundary conditions, unusual inputs, and corner cases.
"""

import pytest
import time

from llm_cost_guard import CostTracker, Budget, BudgetAction
from llm_cost_guard.rate_limit import RateLimit, RateLimiter, SlidingWindowCounter
from llm_cost_guard.budget import BudgetTracker


class TestSlidingWindowEdgeCases:
    """Edge case tests for SlidingWindowCounter."""

    def test_rejects_zero_window_size(self):
        """Test rejection of zero window size."""
        with pytest.raises(ValueError, match="window_size_seconds must be > 0"):
            SlidingWindowCounter(0, 10)

    def test_rejects_negative_window_size(self):
        """Test rejection of negative window size."""
        with pytest.raises(ValueError, match="window_size_seconds must be > 0"):
            SlidingWindowCounter(-1, 10)

    def test_rejects_zero_limit(self):
        """Test rejection of zero limit."""
        with pytest.raises(ValueError, match="limit must be > 0"):
            SlidingWindowCounter(60, 0)

    def test_rejects_negative_limit(self):
        """Test rejection of negative limit."""
        with pytest.raises(ValueError, match="limit must be > 0"):
            SlidingWindowCounter(60, -1)

    def test_handles_very_short_window(self):
        """Test handling of very short window."""
        counter = SlidingWindowCounter(0.1, 5)  # 100ms window
        
        for _ in range(5):
            counter.record()
        
        # Should be at limit
        allowed, count, _ = counter.check()
        assert not allowed
        assert count == 5
        
        # Wait for window to expire
        time.sleep(0.15)
        
        # Should be allowed again
        allowed, count, _ = counter.check()
        assert allowed

    def test_handles_very_large_limit(self):
        """Test handling of very large limit."""
        counter = SlidingWindowCounter(60, 1000000)
        
        # Should allow requests
        allowed, _, _ = counter.check()
        assert allowed


class TestBudgetEdgeCases:
    """Edge case tests for budget tracking."""

    def test_zero_budget_limit(self):
        """Test handling of zero budget limit."""
        tracker = BudgetTracker([
            Budget(name="zero", limit=0.0, action=BudgetAction.WARN)
        ])
        
        # Utilization should be 0 for zero limit
        assert tracker.get_utilization("zero") == 0.0

    def test_negative_budget_limit(self):
        """Test handling of negative budget limit."""
        tracker = BudgetTracker([
            Budget(name="negative", limit=-10.0, action=BudgetAction.WARN)
        ])
        
        # Utilization should be 0 for negative limit
        assert tracker.get_utilization("negative") == 0.0

    def test_very_small_cost(self):
        """Test handling of very small costs."""
        tracker = BudgetTracker([
            Budget(name="test", limit=1.0, action=BudgetAction.WARN)
        ])
        
        # Record many tiny costs
        for _ in range(1000):
            tracker.record_cost(0.0001, {})
        
        # Should accumulate correctly
        spending = tracker.get_spending("test")
        assert abs(spending - 0.1) < 0.001

    def test_very_large_cost(self):
        """Test handling of very large costs."""
        tracker = BudgetTracker([
            Budget(name="test", limit=1000000.0, action=BudgetAction.WARN)
        ])
        
        tracker.record_cost(999999.99, {})
        
        spending = tracker.get_spending("test")
        assert abs(spending - 999999.99) < 0.01

    def test_nonexistent_budget(self):
        """Test queries for nonexistent budget."""
        tracker = BudgetTracker([])
        
        assert tracker.get_spending("nonexistent") == 0.0
        assert tracker.get_remaining("nonexistent") == float("inf")
        assert tracker.get_utilization("nonexistent") == 0.0
        assert tracker.get_budget("nonexistent") is None


class TestCostTrackerEdgeCases:
    """Edge case tests for CostTracker."""

    def test_zero_tokens(self):
        """Test handling of zero token counts."""
        tracker = CostTracker()
        
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
        )
        
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.total_cost == 0.0
        
        tracker.close()

    def test_very_large_token_counts(self):
        """Test handling of very large token counts."""
        tracker = CostTracker()
        
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000000,
            output_tokens=1000000,
        )
        
        assert record.input_tokens == 1000000
        assert record.output_tokens == 1000000
        assert record.total_cost > 0
        
        tracker.close()

    def test_empty_tags(self):
        """Test handling of empty tags dict."""
        tracker = CostTracker()
        
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            tags={},
        )
        
        assert record.tags == {}
        
        tracker.close()

    def test_many_tags(self):
        """Test handling of many tags."""
        tracker = CostTracker()
        
        tags = {f"tag_{i}": f"value_{i}" for i in range(100)}
        
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            tags=tags,
        )
        
        assert len(record.tags) == 100
        
        tracker.close()

    def test_unicode_in_tags(self):
        """Test handling of unicode in tags."""
        tracker = CostTracker()
        
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            tags={"team": "团队", "feature": "功能"},
        )
        
        assert record.tags["team"] == "团队"
        
        tracker.close()


class TestPricingEdgeCases:
    """Edge case tests for pricing calculations."""

    def test_cached_tokens_exceed_input(self):
        """Test when cached tokens exceed input tokens."""
        from llm_cost_guard.pricing.loader import PricingLoader
        
        loader = PricingLoader()
        
        # This should cap cached_tokens and warn
        input_cost, output_cost, total_cost = loader.calculate_cost(
            "openai", "gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=200,  # More than input!
        )
        
        # Should still produce valid result
        assert total_cost >= 0

    def test_zero_cost_model(self):
        """Test handling when model has zero pricing."""
        from llm_cost_guard.pricing.loader import PricingLoader
        
        loader = PricingLoader(pricing_overrides={
            "test/free-model": {
                "input_cost_per_1k": 0.0,
                "output_cost_per_1k": 0.0,
            }
        })
        
        input_cost, output_cost, total_cost = loader.calculate_cost(
            "test", "free-model",
            input_tokens=10000,
            output_tokens=5000,
        )
        
        assert total_cost == 0.0


class TestRateLimiterEdgeCases:
    """Edge case tests for RateLimiter."""

    def test_empty_rate_limits(self):
        """Test with no rate limits configured."""
        limiter = RateLimiter([])
        
        # Should allow all requests
        exceeded = limiter.check()
        assert len(exceeded) == 0
        
        exceeded = limiter.record()
        assert len(exceeded) == 0

    def test_multiple_scopes(self):
        """Test with multiple scope types."""
        limiter = RateLimiter([
            RateLimit(name="global", limit=100, scope="global"),
            RateLimit(name="per_model", limit=10, scope="model"),
            RateLimit(name="per_team", limit=50, scope="tag:team"),
        ])
        
        exceeded = limiter.check(model="gpt-4o", tags={"team": "search"})
        assert len(exceeded) == 0

    def test_unknown_scope(self):
        """Test with unknown scope value."""
        limiter = RateLimiter([
            RateLimit(name="test", limit=10, scope="unknown_scope"),
        ])
        
        # Should fall back to global
        exceeded = limiter.check()
        assert len(exceeded) == 0
