"""
Rate limiting for LLM Cost Guard.
"""

import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

RateLimitPeriod = Literal["second", "minute", "hour"]
RateLimitScope = Literal["global", "model", "provider"]


@dataclass
class RateLimit:
    """Rate limit configuration."""

    name: str
    limit: int
    period: RateLimitPeriod = "minute"
    scope: str = "global"  # "global", "model", "provider", or "tag:key_name"


class SlidingWindowCounter:
    """
    Sliding window rate limiter implementation.

    Thread-safe with memory-bounded storage.
    """

    # Maximum requests to store to prevent memory issues
    MAX_STORED_REQUESTS = 100000

    def __init__(self, window_size_seconds: float, limit: int):
        if window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be > 0")
        if limit <= 0:
            raise ValueError("limit must be > 0")

        self._window_size = window_size_seconds
        self._limit = limit
        self._requests: list[float] = []
        self._lock = threading.Lock()

    def _cleanup(self, now: float) -> None:
        """Remove expired entries and enforce memory bound."""
        cutoff = now - self._window_size
        self._requests = [t for t in self._requests if t > cutoff]

        # Memory safety: truncate if too many requests
        if len(self._requests) > self.MAX_STORED_REQUESTS:
            self._requests = self._requests[-self.MAX_STORED_REQUESTS:]

    def check(self) -> tuple[bool, int, Optional[float]]:
        """
        Check if a request would be allowed.
        Returns (allowed, current_count, retry_after_seconds).
        """
        now = time.time()
        with self._lock:
            self._cleanup(now)
            current = len(self._requests)

            if current >= self._limit:
                # Calculate when the oldest request will expire
                if self._requests:
                    retry_after = self._requests[0] + self._window_size - now
                    return False, current, max(0.0, retry_after)
                return False, current, self._window_size

            return True, current, None

    def record(self) -> bool:
        """
        Record a request. Returns True if allowed, False if rate limited.
        """
        now = time.time()
        with self._lock:
            self._cleanup(now)

            if len(self._requests) >= self._limit:
                return False

            self._requests.append(now)
            return True

    def get_count(self) -> int:
        """Get current request count in the window."""
        now = time.time()
        with self._lock:
            self._cleanup(now)
            return len(self._requests)

    def reset(self) -> None:
        """Reset the counter."""
        with self._lock:
            self._requests = []


class RateLimiter:
    """Manages rate limiting across multiple limits and scopes."""

    def __init__(self, rate_limits: Optional[list[RateLimit]] = None):
        self._rate_limits = rate_limits or []
        # Key: (limit_name, scope_value) -> SlidingWindowCounter
        self._counters: dict[tuple[str, str], SlidingWindowCounter] = {}
        self._lock = threading.Lock()

    def _get_period_seconds(self, period: RateLimitPeriod) -> float:
        """Get period in seconds."""
        periods = {
            "second": 1.0,
            "minute": 60.0,
            "hour": 3600.0,
        }
        return periods.get(period, 60.0)

    def _get_scope_key(
        self,
        rate_limit: RateLimit,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """Get the scope key for a rate limit."""
        scope = rate_limit.scope
        tags = tags or {}

        if scope == "global":
            return "global"
        elif scope == "model":
            return f"model:{model or 'unknown'}"
        elif scope == "provider":
            return f"provider:{provider or 'unknown'}"
        elif scope.startswith("tag:"):
            tag_key = scope[4:]
            tag_value = tags.get(tag_key, "unknown")
            return f"tag:{tag_key}:{tag_value}"
        return "global"

    def _get_counter(
        self,
        rate_limit: RateLimit,
        scope_key: str,
    ) -> SlidingWindowCounter:
        """Get or create a counter for a rate limit and scope."""
        key = (rate_limit.name, scope_key)

        with self._lock:
            if key not in self._counters:
                window_size = self._get_period_seconds(rate_limit.period)
                self._counters[key] = SlidingWindowCounter(window_size, rate_limit.limit)
            return self._counters[key]

    def add_rate_limit(self, rate_limit: RateLimit) -> None:
        """Add a new rate limit."""
        self._rate_limits.append(rate_limit)

    def remove_rate_limit(self, name: str) -> bool:
        """Remove a rate limit by name."""
        for i, rl in enumerate(self._rate_limits):
            if rl.name == name:
                del self._rate_limits[i]
                # Remove associated counters
                with self._lock:
                    keys_to_remove = [k for k in self._counters if k[0] == name]
                    for k in keys_to_remove:
                        del self._counters[k]
                return True
        return False

    def get_rate_limit(self, name: str) -> Optional[RateLimit]:
        """Get a rate limit by name."""
        for rl in self._rate_limits:
            if rl.name == name:
                return rl
        return None

    def check(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> list[tuple[RateLimit, int, Optional[float]]]:
        """
        Check all rate limits.
        Returns list of (rate_limit, current_count, retry_after) for exceeded limits.
        """
        exceeded = []

        for rate_limit in self._rate_limits:
            scope_key = self._get_scope_key(rate_limit, model, provider, tags)
            counter = self._get_counter(rate_limit, scope_key)
            allowed, current, retry_after = counter.check()

            if not allowed:
                exceeded.append((rate_limit, current, retry_after))

        return exceeded

    def record(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> list[tuple[RateLimit, int, Optional[float]]]:
        """
        Record a request against all applicable rate limits.
        Returns list of (rate_limit, current_count, retry_after) for limits that are now exceeded.
        """
        exceeded = []

        for rate_limit in self._rate_limits:
            scope_key = self._get_scope_key(rate_limit, model, provider, tags)
            counter = self._get_counter(rate_limit, scope_key)

            if not counter.record():
                _, current, retry_after = counter.check()
                exceeded.append((rate_limit, current, retry_after))

        return exceeded

    def get_remaining(
        self,
        name: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> int:
        """Get remaining requests for a rate limit."""
        rate_limit = self.get_rate_limit(name)
        if rate_limit is None:
            return 0

        scope_key = self._get_scope_key(rate_limit, model, provider, tags)
        counter = self._get_counter(rate_limit, scope_key)
        return max(0, rate_limit.limit - counter.get_count())

    def reset(self, name: Optional[str] = None) -> None:
        """Reset counters for a specific rate limit or all rate limits."""
        with self._lock:
            if name:
                keys_to_reset = [k for k in self._counters if k[0] == name]
                for k in keys_to_reset:
                    self._counters[k].reset()
            else:
                for counter in self._counters.values():
                    counter.reset()
