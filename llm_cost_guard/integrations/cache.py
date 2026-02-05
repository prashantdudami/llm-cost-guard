"""
Cache integration for LLM Cost Guard.
"""

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CacheTracker:
    """
    Tracks cache hits and savings for cached LLM calls.

    Usage:
        from llm_cost_guard import CostTracker
        from llm_cost_guard.integrations.cache import CacheTracker

        tracker = CostTracker()
        cache_tracker = CacheTracker(tracker)

        @cache_tracker.track
        @your_cache_decorator
        def cached_llm_call(prompt):
            return llm.invoke(prompt)
    """

    def __init__(
        self,
        tracker: Any,  # CostTracker
        default_tags: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the cache tracker.

        Args:
            tracker: CostTracker instance
            default_tags: Default tags to apply to all tracked calls
        """
        self._tracker = tracker
        self._default_tags = default_tags or {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._estimated_savings = 0.0

    def track(
        self,
        func: Optional[F] = None,
        *,
        tags: Optional[dict[str, str]] = None,
        cache_indicator: str = "_from_cache",
    ) -> F:
        """
        Decorator to track cache hits and savings.

        The decorated function should set a `_from_cache` attribute on
        the result if it came from cache, or return a tuple (result, from_cache).

        Args:
            func: Function to decorate
            tags: Additional tags
            cache_indicator: Attribute name to check for cache hit

        Returns:
            Decorated function
        """

        def decorator(f: F) -> F:
            @functools.wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = f(*args, **kwargs)

                # Check if result came from cache
                from_cache = False

                # Check for tuple return
                if isinstance(result, tuple) and len(result) == 2:
                    actual_result, from_cache = result
                    result = actual_result

                # Check for attribute on result
                elif hasattr(result, cache_indicator):
                    from_cache = getattr(result, cache_indicator, False)

                # Update cache stats
                if from_cache:
                    self._cache_hits += 1
                else:
                    self._cache_misses += 1

                return result

            return wrapper  # type: ignore

        if func is not None:
            return decorator(func)
        return decorator  # type: ignore

    def record_cache_hit(
        self,
        estimated_cost: float,
        provider: str = "unknown",
        model: str = "unknown",
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Manually record a cache hit with estimated savings.

        Args:
            estimated_cost: Estimated cost that was saved
            provider: Provider name
            model: Model name
            tags: Attribution tags
        """
        self._cache_hits += 1
        self._estimated_savings += estimated_cost

        # Record in the main tracker as a zero-cost call
        all_tags = dict(self._default_tags)
        if tags:
            all_tags.update(tags)
        all_tags["cache_hit"] = "true"

        self._tracker.record(
            provider=provider,
            model=model,
            input_tokens=0,
            output_tokens=0,
            tags=all_tags,
            success=True,
            metadata={"cache_savings": estimated_cost},
        )

    def record_cache_miss(
        self,
        provider: str = "unknown",
        model: str = "unknown",
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Manually record a cache miss.

        Args:
            provider: Provider name
            model: Model name
            tags: Attribution tags
        """
        self._cache_misses += 1

    @property
    def cache_hits(self) -> int:
        """Get total cache hits."""
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """Get total cache misses."""
        return self._cache_misses

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0)."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    @property
    def estimated_savings(self) -> float:
        """Get estimated cost savings from cache hits."""
        return self._estimated_savings

    def reset(self) -> None:
        """Reset cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._estimated_savings = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "estimated_savings": self._estimated_savings,
        }
