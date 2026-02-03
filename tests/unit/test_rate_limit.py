"""
Unit tests for rate limiting.
"""

import pytest
import time

from llm_cost_guard.rate_limit import RateLimit, RateLimiter, SlidingWindowCounter


class TestSlidingWindowCounter:
    """Tests for SlidingWindowCounter."""

    def test_counter_basic(self):
        """Test basic counter functionality."""
        counter = SlidingWindowCounter(window_size_seconds=1.0, limit=5)

        # Should allow 5 requests
        for _ in range(5):
            assert counter.record() is True

        # 6th request should be denied
        allowed, count, retry_after = counter.check()
        assert allowed is False
        assert count == 5

    def test_counter_window_expiry(self):
        """Test that old requests expire."""
        counter = SlidingWindowCounter(window_size_seconds=0.1, limit=2)

        # Use up the limit
        assert counter.record() is True
        assert counter.record() is True
        assert counter.record() is False

        # Wait for window to expire
        time.sleep(0.15)

        # Should allow again
        assert counter.record() is True

    def test_counter_get_count(self):
        """Test get_count method."""
        counter = SlidingWindowCounter(window_size_seconds=1.0, limit=10)

        assert counter.get_count() == 0

        counter.record()
        assert counter.get_count() == 1

        counter.record()
        counter.record()
        assert counter.get_count() == 3

    def test_counter_reset(self):
        """Test reset method."""
        counter = SlidingWindowCounter(window_size_seconds=1.0, limit=10)

        counter.record()
        counter.record()
        assert counter.get_count() == 2

        counter.reset()
        assert counter.get_count() == 0


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_limiter_basic(self):
        """Test basic rate limiter."""
        limiter = RateLimiter([
            RateLimit(name="global", limit=5, period="minute", scope="global")
        ])

        # Record 5 requests - should all succeed
        for _ in range(5):
            exceeded = limiter.record()
            assert len(exceeded) == 0

        # 6th request should fail
        exceeded = limiter.check()
        assert len(exceeded) == 1
        assert exceeded[0][0].name == "global"

    def test_limiter_model_scope(self):
        """Test rate limiter with model scope."""
        limiter = RateLimiter([
            RateLimit(name="model-limit", limit=2, period="minute", scope="model")
        ])

        # Record requests for gpt-4o
        limiter.record(model="gpt-4o")
        limiter.record(model="gpt-4o")

        # gpt-4o should be limited
        exceeded = limiter.check(model="gpt-4o")
        assert len(exceeded) == 1

        # gpt-3.5-turbo should still be allowed
        exceeded = limiter.check(model="gpt-3.5-turbo")
        assert len(exceeded) == 0

    def test_limiter_provider_scope(self):
        """Test rate limiter with provider scope."""
        limiter = RateLimiter([
            RateLimit(name="provider-limit", limit=3, period="minute", scope="provider")
        ])

        # Fill up OpenAI quota
        limiter.record(provider="openai")
        limiter.record(provider="openai")
        limiter.record(provider="openai")

        # OpenAI should be limited
        exceeded = limiter.check(provider="openai")
        assert len(exceeded) == 1

        # Anthropic should be allowed
        exceeded = limiter.check(provider="anthropic")
        assert len(exceeded) == 0

    def test_limiter_tag_scope(self):
        """Test rate limiter with tag scope."""
        limiter = RateLimiter([
            RateLimit(name="user-limit", limit=2, period="minute", scope="tag:user_id")
        ])

        # Fill up user1's quota
        limiter.record(tags={"user_id": "user1"})
        limiter.record(tags={"user_id": "user1"})

        # user1 should be limited
        exceeded = limiter.check(tags={"user_id": "user1"})
        assert len(exceeded) == 1

        # user2 should be allowed
        exceeded = limiter.check(tags={"user_id": "user2"})
        assert len(exceeded) == 0

    def test_limiter_get_remaining(self):
        """Test get_remaining method."""
        limiter = RateLimiter([
            RateLimit(name="test", limit=5, period="minute", scope="global")
        ])

        assert limiter.get_remaining("test") == 5

        limiter.record()
        assert limiter.get_remaining("test") == 4

        limiter.record()
        limiter.record()
        assert limiter.get_remaining("test") == 2

    def test_limiter_add_remove(self):
        """Test adding and removing rate limits."""
        limiter = RateLimiter()

        assert limiter.get_rate_limit("test") is None

        limiter.add_rate_limit(RateLimit(name="test", limit=10, period="minute", scope="global"))
        assert limiter.get_rate_limit("test") is not None

        limiter.remove_rate_limit("test")
        assert limiter.get_rate_limit("test") is None

    def test_limiter_multiple_limits(self):
        """Test multiple rate limits."""
        limiter = RateLimiter([
            RateLimit(name="global", limit=10, period="minute", scope="global"),
            RateLimit(name="model", limit=3, period="minute", scope="model"),
        ])

        # Fill up model limit for gpt-4o
        limiter.record(model="gpt-4o")
        limiter.record(model="gpt-4o")
        limiter.record(model="gpt-4o")

        # Model limit should be exceeded, global should not
        exceeded = limiter.check(model="gpt-4o")
        assert len(exceeded) == 1
        assert exceeded[0][0].name == "model"

    def test_limiter_reset(self):
        """Test resetting rate limits."""
        limiter = RateLimiter([
            RateLimit(name="test", limit=5, period="minute", scope="global")
        ])

        limiter.record()
        limiter.record()
        assert limiter.get_remaining("test") == 3

        limiter.reset("test")
        assert limiter.get_remaining("test") == 5
