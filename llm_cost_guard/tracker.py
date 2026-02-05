"""
Main CostTracker class for LLM Cost Guard.
"""

import asyncio
import functools
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Literal, Optional, TypeVar, Union

from llm_cost_guard.audit import AuditBackend, AuditLogger, LoggingAuditBackend
from llm_cost_guard.backends import Backend, MemoryBackend, get_backend
from llm_cost_guard.budget import Budget, BudgetAction, BudgetTracker
from llm_cost_guard.exceptions import (
    BudgetExceededError,
    RateLimitExceededError,
    TrackingUnavailableError,
)
from llm_cost_guard.models import CostRecord, CostReport, HealthStatus, ModelType
from llm_cost_guard.pricing.loader import PricingLoader
from llm_cost_guard.providers import detect_provider, get_provider
from llm_cost_guard.rate_limit import RateLimit, RateLimiter
from llm_cost_guard.span import Span, get_current_span

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CostTracker:
    """
    Main entry point for cost tracking.

    Provides decorator-based and context manager tracking for LLM API calls.
    """

    def __init__(
        self,
        budgets: Optional[list[Budget]] = None,
        rate_limits: Optional[list[RateLimit]] = None,
        backend: str = "memory",
        auto_detect_provider: bool = True,
        pricing_update: bool = True,
        pricing_overrides: Optional[dict[str, dict[str, Any]]] = None,
        on_tracking_failure: Literal["block", "allow", "fallback"] = "allow",
        store_prompts: bool = False,
        track_failed_calls: bool = True,
        track_cache_savings: bool = True,
        max_unique_tag_values: int = 1000,
        budget_mode: Literal["local", "distributed"] = "local",
        streaming_budget_mode: Literal["estimate", "actual"] = "actual",
        streaming_max_output_estimate: int = 4096,
        audit_enabled: bool = True,
        audit_backend: Optional[AuditBackend] = None,
        **backend_kwargs: Any,
    ):
        """
        Initialize the CostTracker.

        Args:
            budgets: List of budget configurations
            rate_limits: List of rate limit configurations
            backend: Backend URL (memory, sqlite:///, postgresql://, redis://)
            auto_detect_provider: Automatically detect provider from model name
            pricing_update: Check for pricing updates on startup
            pricing_overrides: Custom pricing for models (e.g., negotiated rates)
            on_tracking_failure: Action when tracking fails (block/allow/fallback)
            store_prompts: Store prompts in records (default: False for security)
            track_failed_calls: Track costs for failed API calls
            track_cache_savings: Track cache hit savings
            max_unique_tag_values: Maximum unique values per tag key
            budget_mode: Budget enforcement mode (local or distributed)
            streaming_budget_mode: How to handle streaming budgets
            streaming_max_output_estimate: Max output tokens to estimate for streaming
            audit_enabled: Enable audit logging for compliance
            audit_backend: Custom audit backend (defaults to logging)
        """
        self._auto_detect_provider = auto_detect_provider
        self._on_tracking_failure = on_tracking_failure
        self._store_prompts = store_prompts
        self._track_failed_calls = track_failed_calls
        self._track_cache_savings = track_cache_savings
        self._max_unique_tag_values = max_unique_tag_values
        self._budget_mode = budget_mode
        self._streaming_budget_mode = streaming_budget_mode
        self._streaming_max_output_estimate = streaming_max_output_estimate

        # Graceful degradation metrics
        self._metrics = {
            "backend_failures": 0,
            "fallback_activations": 0,
            "budget_checks": 0,
            "budget_exceeded_count": 0,
            "rate_limit_exceeded_count": 0,
            "tracking_errors": 0,
        }
        self._metrics_lock = threading.Lock()

        # Initialize audit logging
        self._audit = AuditLogger(
            backend=audit_backend or LoggingAuditBackend(),
            enabled=audit_enabled,
        )

        # Initialize backend
        self._backend_url = backend
        self._fallback_backend: Optional[MemoryBackend] = None
        self._using_fallback = False
        try:
            self._backend: Backend = get_backend(backend, **backend_kwargs)
        except Exception as e:
            self._increment_metric("backend_failures")
            if on_tracking_failure == "block":
                raise TrackingUnavailableError(f"Failed to initialize backend: {e}", backend)
            elif on_tracking_failure == "fallback":
                logger.warning(f"Failed to initialize backend {backend}, using memory fallback: {e}")
                self._backend = MemoryBackend()
                self._fallback_backend = self._backend
                self._using_fallback = True
                self._increment_metric("fallback_activations")
                self._audit.log_fallback_activated(backend, "memory", str(e))
            else:
                logger.warning(f"Failed to initialize backend {backend}: {e}")
                self._backend = MemoryBackend()
                self._audit.log_tracking_failure(str(e), backend, "allow")

        # Initialize pricing
        self._pricing = PricingLoader(pricing_overrides=pricing_overrides)

        # Initialize budget tracking
        self._budget_tracker = BudgetTracker(budgets)

        # Log budget creation for audit
        for budget in (budgets or []):
            self._audit.log_budget_created(
                budget.name,
                budget.limit,
                budget.period,
                budget.action.value,
            )

        # Initialize rate limiting
        self._rate_limiter = RateLimiter(rate_limits)

        # Tag cardinality tracking
        self._tag_values: dict[str, set] = {}
        self._tag_lock = threading.Lock()

        # Last call tracking
        self._last_record: Optional[CostRecord] = None
        self._lock = threading.Lock()

    def _increment_metric(self, metric: str, amount: int = 1) -> None:
        """Thread-safe metric increment."""
        with self._metrics_lock:
            self._metrics[metric] = self._metrics.get(metric, 0) + amount

    def track(
        self,
        func: Optional[F] = None,
        *,
        tags: Optional[dict[str, str]] = None,
        streaming: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Union[F, Callable[[F], F]]:
        """
        Decorator to track LLM call costs.

        Can be used with or without arguments:
            @tracker.track
            def my_call(): ...

            @tracker.track(tags={"team": "search"})
            def my_call(): ...

        Args:
            func: Function to decorate (when used without arguments)
            tags: Tags for attribution
            streaming: Whether the function returns a streaming response
            provider: Override provider detection
            model: Override model detection

        Returns:
            Decorated function
        """

        def decorator(f: F) -> F:
            if asyncio.iscoroutinefunction(f):
                return self._wrap_async(f, tags, streaming, provider, model)  # type: ignore
            else:
                return self._wrap_sync(f, tags, streaming, provider, model)  # type: ignore

        if func is not None:
            return decorator(func)
        return decorator

    def _wrap_sync(
        self,
        func: F,
        tags: Optional[dict[str, str]],
        streaming: bool,
        provider_override: Optional[str],
        model_override: Optional[str],
    ) -> F:
        """Wrap a synchronous function for tracking."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            success = True
            error_type = None
            response = None

            try:
                response = func(*args, **kwargs)
                return response
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                latency_ms = int((time.time() - start_time) * 1000)

                if response is not None or (not success and self._track_failed_calls):
                    try:
                        self._record_call(
                            response=response,
                            tags=tags,
                            success=success,
                            error_type=error_type,
                            latency_ms=latency_ms,
                            provider_override=provider_override,
                            model_override=model_override,
                        )
                    except Exception as e:
                        self._handle_tracking_error(e)

        return wrapper  # type: ignore

    def _wrap_async(
        self,
        func: F,
        tags: Optional[dict[str, str]],
        streaming: bool,
        provider_override: Optional[str],
        model_override: Optional[str],
    ) -> F:
        """Wrap an asynchronous function for tracking."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            success = True
            error_type = None
            response = None

            try:
                response = await func(*args, **kwargs)
                return response
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                latency_ms = int((time.time() - start_time) * 1000)

                if response is not None or (not success and self._track_failed_calls):
                    try:
                        self._record_call(
                            response=response,
                            tags=tags,
                            success=success,
                            error_type=error_type,
                            latency_ms=latency_ms,
                            provider_override=provider_override,
                            model_override=model_override,
                        )
                    except Exception as e:
                        self._handle_tracking_error(e)

        return wrapper  # type: ignore

    @contextmanager
    def track_context(
        self,
        tags: Optional[dict[str, str]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Context manager for tracking LLM calls.

        Usage:
            with tracker.track_context(tags={"feature": "search"}):
                response = openai.chat.completions.create(...)

        Note: This context manager doesn't automatically extract usage from
        responses. Use the decorator or manual recording for automatic tracking.
        """
        time.time()
        tags = tags or {}

        try:
            yield
        finally:
            pass  # Context manager for grouping, actual tracking via decorator or record()

    def span(
        self,
        name: str,
        tags: Optional[dict[str, str]] = None,
    ) -> Span:
        """
        Create a tracking span for grouping multiple LLM calls.

        Usage:
            with tracker.span("rag_pipeline", tags={"user": "123"}) as span:
                # Multiple LLM calls here
                result = agent.run(query)
                print(span.total_cost)

        Args:
            name: Name of the span
            tags: Tags for the span

        Returns:
            Span context manager
        """
        return Span(name=name, tags=tags or {})

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tags: Optional[dict[str, str]] = None,
        success: bool = True,
        error_type: Optional[str] = None,
        latency_ms: int = 0,
        cached_tokens: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CostRecord:
        """
        Manually record an LLM call.

        Use this for custom integrations or when automatic tracking isn't available.

        Args:
            provider: Provider name (openai, anthropic, bedrock)
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tags: Attribution tags
            success: Whether the call succeeded
            error_type: Error type if call failed
            latency_ms: Call latency in milliseconds
            cached_tokens: Number of cached input tokens
            metadata: Additional metadata (high-cardinality data)

        Returns:
            The created CostRecord

        Raises:
            ValueError: If provider or model names are invalid
        """
        # Input validation
        self._validate_identifier(provider, "provider")
        self._validate_identifier(model, "model")

        if input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        if output_tokens < 0:
            raise ValueError("output_tokens must be >= 0")
        if cached_tokens < 0:
            raise ValueError("cached_tokens must be >= 0")
        if latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")

        tags = tags or {}
        metadata = metadata or {}

        # Validate tag cardinality
        self._check_tag_cardinality(tags)

        # Calculate cost
        input_cost, output_cost, total_cost = self._pricing.calculate_cost(
            provider, model, input_tokens, output_tokens, cached_tokens
        )

        # Calculate cache savings
        cache_savings = 0.0
        if cached_tokens > 0 and self._track_cache_savings:
            pricing = self._pricing.get_pricing(provider, model)
            if pricing.cached_input_cost_per_1k is not None:
                cache_savings = (cached_tokens / 1000) * (
                    pricing.input_cost_per_1k - pricing.cached_input_cost_per_1k
                )

        # Check budgets
        self._increment_metric("budget_checks")
        exceeded = self._budget_tracker.check_budget(total_cost, tags)
        for budget, action in exceeded:
            current_spending = self._budget_tracker.get_spending(budget.name)

            # Log to audit
            if action == BudgetAction.WARN:
                self._audit.log_budget_warning(
                    budget.name,
                    current_spending,
                    budget.limit,
                    current_spending / budget.limit,
                )
            elif action == BudgetAction.BLOCK:
                self._increment_metric("budget_exceeded_count")
                self._audit.log_budget_exceeded(
                    budget.name,
                    current_spending,
                    budget.limit,
                    "blocked",
                )
                raise BudgetExceededError(
                    f"Budget '{budget.name}' would be exceeded",
                    budget=budget,
                    current=current_spending,
                    limit=budget.limit,
                )

        # Check rate limits
        rate_exceeded = self._rate_limiter.check(model=model, provider=provider, tags=tags)
        if rate_exceeded:
            limit, current, retry_after = rate_exceeded[0]
            self._increment_metric("rate_limit_exceeded_count")
            self._audit.log_rate_limit_exceeded(
                limit.name,
                current,
                limit.limit,
                retry_after,
            )
            raise RateLimitExceededError(
                f"Rate limit '{limit.name}' exceeded",
                limit_name=limit.name,
                current=current,
                limit=limit.limit,
                retry_after_seconds=retry_after,
            )

        # Create record
        record = CostRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            model_type=ModelType.CHAT,  # Default, can be enhanced
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            latency_ms=latency_ms,
            tags=tags,
            metadata=metadata,
            success=success,
            error_type=error_type,
            cached=cached_tokens > 0,
            cache_savings=cache_savings,
            span_id=get_current_span().span_id if get_current_span() else None,
        )

        # Save to backend
        try:
            self._backend.save_record(record)
        except Exception as e:
            self._handle_tracking_error(e)

        # Record against budgets
        self._budget_tracker.record_cost(total_cost, tags)

        # Record rate limit usage
        self._rate_limiter.record(model=model, provider=provider, tags=tags)

        # Update last record
        with self._lock:
            self._last_record = record

        # Record in current span if any
        current_span = get_current_span()
        if current_span:
            current_span.record_call(
                cost=total_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                record=record,
            )

        return record

    def _record_call(
        self,
        response: Any,
        tags: Optional[dict[str, str]],
        success: bool,
        error_type: Optional[str],
        latency_ms: int,
        provider_override: Optional[str],
        model_override: Optional[str],
    ) -> Optional[CostRecord]:
        """Record a call from a wrapped function."""
        if response is None:
            return None

        # Detect provider and model
        if model_override:
            model = model_override
            provider = provider_override or (
                detect_provider(model) if self._auto_detect_provider else "unknown"
            )
        else:
            # Try to extract from response
            provider = provider_override or "openai"  # Default
            model = "unknown"

            # Try to detect and extract
            try:
                if self._auto_detect_provider:
                    # Try OpenAI-style response
                    if hasattr(response, "model"):
                        model = response.model
                        provider = detect_provider(model)
                    elif isinstance(response, dict) and "model" in response:
                        model = response["model"]
                        provider = detect_provider(model)
            except Exception:
                pass

        # Get provider handler
        try:
            provider_handler = get_provider(provider)
        except ValueError:
            logger.warning(f"Unknown provider {provider}, skipping cost tracking")
            return None

        # Extract usage
        usage = provider_handler.extract_usage(response)
        if model == "unknown":
            model = provider_handler.extract_model(response)

        # Record
        return self.record(
            provider=provider,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tags=tags,
            success=success,
            error_type=error_type,
            latency_ms=latency_ms,
            cached_tokens=usage.cached_tokens,
        )

    def _handle_tracking_error(self, error: Exception) -> None:
        """
        Handle errors during tracking based on configuration.

        Behavior depends on on_tracking_failure setting:
        - "block": Raises TrackingUnavailableError
        - "fallback": Switches to in-memory backend
        - "allow": Logs warning and continues

        Args:
            error: The exception that occurred
        """
        self._increment_metric("tracking_errors")
        self._increment_metric("backend_failures")

        error_context = f"[backend={self._backend_url}] {type(error).__name__}: {error}"

        if self._on_tracking_failure == "block":
            self._audit.log_tracking_failure(error_context, self._backend_url, "blocked")
            logger.error(f"Tracking blocked due to error: {error_context}")
            raise TrackingUnavailableError(
                f"Cost tracking unavailable: {error_context}",
                self._backend_url
            )
        elif self._on_tracking_failure == "fallback":
            logger.warning(
                f"Tracking error, switching to fallback backend: {error_context}"
            )
            if self._fallback_backend is None:
                self._fallback_backend = MemoryBackend()
                self._using_fallback = True
                self._increment_metric("fallback_activations")
                self._audit.log_fallback_activated(
                    self._backend_url, "memory", error_context
                )
        else:
            logger.warning(f"Tracking error (continuing): {error_context}")
            self._audit.log_tracking_failure(error_context, self._backend_url, "allowed")

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        """
        Validate that a value is a valid identifier.

        Security: Prevents injection attacks via provider/model names.

        Args:
            value: The value to validate
            name: Name of the parameter (for error messages)

        Raises:
            ValueError: If the value is not valid
        """
        import re

        if not value or not isinstance(value, str):
            raise ValueError(f"{name} must be a non-empty string")

        if len(value) > 256:
            raise ValueError(f"{name} must be <= 256 characters")

        # Allow common model name patterns: alphanumeric, dash, underscore, dot, colon, slash
        # Examples: gpt-4o, claude-3.5-sonnet, anthropic.claude-v2:1, bedrock/claude
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.:/]*$', value):
            raise ValueError(
                f"Invalid {name}: must start with alphanumeric and contain only "
                f"alphanumeric, dash, underscore, dot, colon, or slash"
            )

    def _check_tag_cardinality(self, tags: dict[str, str]) -> None:
        """Check and track tag cardinality."""
        with self._tag_lock:
            for key, value in tags.items():
                if key not in self._tag_values:
                    self._tag_values[key] = set()

                self._tag_values[key].add(value)

                if len(self._tag_values[key]) > self._max_unique_tag_values:
                    logger.warning(
                        f"Tag '{key}' has exceeded cardinality limit "
                        f"({len(self._tag_values[key])} > {self._max_unique_tag_values})"
                    )

    def last_call(self) -> Optional[CostRecord]:
        """Get the last recorded call."""
        with self._lock:
            return self._last_record

    def get_costs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        group_by: Optional[list[str]] = None,
    ) -> CostReport:
        """
        Query tracked costs.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            tags: Filter by tags
            group_by: Group results by fields (provider, model, or tag keys)

        Returns:
            CostReport with aggregated data
        """
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        return self._backend.get_report(
            start_date=start,
            end_date=end,
            tags=tags,
            group_by=group_by,
        )

    def daily_report(self) -> CostReport:
        """Get a report for today."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self._backend.get_report(start_date=today)

    def report_by_model(self, period: str = "day") -> CostReport:
        """Get a report grouped by model."""
        start = self._get_period_start(period)
        return self._backend.get_report(start_date=start, group_by=["model"])

    def trend_analysis(
        self,
        metric: str = "cost",
        granularity: str = "hour",
        last_n_days: int = 7,
    ) -> dict[str, Any]:
        """Get trend analysis for a metric."""
        # This is a simplified implementation
        from datetime import timedelta

        end = datetime.now()
        start = end - timedelta(days=last_n_days)

        records = self._backend.get_records(start_date=start, end_date=end)

        # Group by time bucket
        buckets: dict[str, float] = {}
        for record in records:
            if granularity == "hour":
                bucket_key = record.timestamp.strftime("%Y-%m-%d %H:00")
            elif granularity == "day":
                bucket_key = record.timestamp.strftime("%Y-%m-%d")
            else:
                bucket_key = record.timestamp.strftime("%Y-%m-%d %H:00")

            if bucket_key not in buckets:
                buckets[bucket_key] = 0.0

            if metric == "cost":
                buckets[bucket_key] += record.total_cost
            elif metric == "tokens":
                buckets[bucket_key] += record.input_tokens + record.output_tokens
            elif metric == "calls":
                buckets[bucket_key] += 1

        return {
            "metric": metric,
            "granularity": granularity,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "data": buckets,
        }

    def _get_period_start(self, period: str) -> datetime:
        """Get the start datetime for a period."""
        from datetime import timedelta

        now = datetime.now()

        if period == "day":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start - timedelta(days=now.weekday())
        elif period == "month":
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def to_dataframe(self):
        """Export records to a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame export. Install with: pip install pandas")

        records = self._backend.get_records()

        data = []
        for r in records:
            row = {
                "timestamp": r.timestamp,
                "provider": r.provider,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_cost": r.total_cost,
                "latency_ms": r.latency_ms,
                "success": r.success,
                "cached": r.cached,
            }
            # Add tags as columns
            for key, value in r.tags.items():
                row[f"tag_{key}"] = value
            data.append(row)

        return pd.DataFrame(data)

    def health_check(self) -> HealthStatus:
        """Check tracker and backend health."""
        errors = []

        # Check backend
        backend_connected = False
        try:
            backend_connected = self._backend.health_check()
        except Exception as e:
            errors.append(f"Backend health check failed: {e}")

        # Check if using fallback
        if self._using_fallback:
            errors.append("Using fallback backend (primary unavailable)")

        # Check pricing freshness
        pricing_fresh = not self._pricing.is_stale
        if self._pricing.is_stale:
            errors.append("Pricing data is stale")

        # Get last record time
        last_record_time = None
        with self._lock:
            if self._last_record:
                last_record_time = self._last_record.timestamp

        return HealthStatus(
            healthy=backend_connected and pricing_fresh and len(errors) == 0 and not self._using_fallback,
            backend_connected=backend_connected,
            pricing_fresh=pricing_fresh,
            last_record_time=last_record_time,
            pending_records=0,
            errors=errors,
            pricing_version=str(self._pricing.pricing_version),
            pricing_last_updated=self._pricing.last_updated,
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        Get tracker metrics for observability.

        Returns metrics for:
        - backend_failures: Number of backend operation failures
        - fallback_activations: Number of times fallback was activated
        - budget_checks: Total budget checks performed
        - budget_exceeded_count: Number of budget exceeded events
        - rate_limit_exceeded_count: Number of rate limit exceeded events
        - tracking_errors: Total tracking errors
        - using_fallback: Whether currently using fallback backend
        """
        with self._metrics_lock:
            metrics = dict(self._metrics)

        metrics["using_fallback"] = self._using_fallback
        metrics["backend_url"] = self._backend_url

        # Add backend-specific metrics if available
        if hasattr(self._backend, "get_metrics"):
            metrics["backend_metrics"] = self._backend.get_metrics()

        return metrics

    @property
    def audit(self) -> AuditLogger:
        """Get the audit logger for querying audit events."""
        return self._audit

    def on_budget_warning(self, callback: Callable[[Budget, float], None]) -> None:
        """Register a callback for budget warnings."""
        self._budget_tracker.on_warning(callback)

    def on_budget_exceeded(self, callback: Callable[[Budget], None]) -> None:
        """Register a callback for budget exceeded events."""
        self._budget_tracker.on_exceeded(callback)

    def get_budget(self, name: str) -> Optional[Budget]:
        """Get a budget by name."""
        return self._budget_tracker.get_budget(name)

    def get_budget_utilization(self, name: str) -> float:
        """Get budget utilization percentage."""
        return self._budget_tracker.get_utilization(name)

    def reset_budget(self, name: Optional[str] = None) -> None:
        """Reset a budget or all budgets."""
        self._budget_tracker.reset(name)

    @property
    def pricing_last_updated(self) -> Optional[datetime]:
        """Get when pricing was last updated."""
        return self._pricing.last_updated

    @property
    def pricing_version(self) -> dict[str, str]:
        """Get pricing versions for all providers."""
        return self._pricing.pricing_version

    @property
    def pricing_is_stale(self) -> bool:
        """Check if pricing is stale."""
        return self._pricing.is_stale

    def close(self) -> None:
        """Close the tracker and backend."""
        self._backend.close()
        if self._fallback_backend:
            self._fallback_backend.close()

    def __enter__(self) -> "CostTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
