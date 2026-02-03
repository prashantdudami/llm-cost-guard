"""
Wrapped Anthropic client with automatic cost tracking.
"""

import time
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_cost_guard import CostTracker


class TrackedAnthropic:
    """
    Anthropic client wrapper with automatic cost tracking.

    Usage:
        from llm_cost_guard import CostTracker
        from llm_cost_guard.clients import TrackedAnthropic

        tracker = CostTracker()
        client = TrackedAnthropic(tracker=tracker)

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Cost is automatically tracked
    """

    def __init__(
        self,
        tracker: "CostTracker",
        client: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
        **anthropic_kwargs: Any,
    ):
        """
        Initialize the tracked Anthropic client.

        Args:
            tracker: CostTracker instance
            client: Optional existing Anthropic client to wrap
            tags: Default tags for all calls
            **anthropic_kwargs: Arguments to pass to Anthropic client
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic is required for this client. "
                "Install with: pip install llm-cost-guard[anthropic]"
            )

        self._tracker = tracker
        self._default_tags = tags or {}
        self._client = client or Anthropic(**anthropic_kwargs)

        # Create wrapped interface
        self.messages = _TrackedMessages(self._client.messages, self._tracker, self._default_tags)

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _TrackedMessages:
    """Wrapped messages API."""

    def __init__(self, messages, tracker: "CostTracker", default_tags: Dict[str, str]):
        self._messages = messages
        self._tracker = tracker
        self._default_tags = default_tags

    def create(
        self,
        *,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a message with tracking."""
        start_time = time.time()
        success = True
        error_type = None
        response = None

        try:
            response = self._messages.create(**kwargs)
            return response
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            if response is not None:
                self._record_response(
                    response, kwargs.get("model"), tags, success, error_type, latency_ms
                )

    def _record_response(
        self,
        response: Any,
        model_hint: Optional[str],
        tags: Optional[Dict[str, str]],
        success: bool,
        error_type: Optional[str],
        latency_ms: int,
    ) -> None:
        """Record the response with the tracker."""
        from llm_cost_guard.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        usage = provider.extract_usage(response)
        model = provider.extract_model(response)

        if model == "unknown" and model_hint:
            model = model_hint

        all_tags = dict(self._default_tags)
        if tags:
            all_tags.update(tags)

        self._tracker.record(
            provider="anthropic",
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tags=all_tags,
            success=success,
            error_type=error_type,
            latency_ms=latency_ms,
            cached_tokens=usage.cached_tokens,
        )

    def stream(
        self,
        *,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a streaming message with tracking."""
        # For streaming, we wrap the stream to track after completion
        start_time = time.time()

        stream = self._messages.stream(**kwargs)

        # Wrap in a tracking context
        return _TrackedStream(
            stream,
            self._tracker,
            self._default_tags,
            tags,
            kwargs.get("model"),
            start_time,
        )


class _TrackedStream:
    """Wrapper for streaming responses."""

    def __init__(
        self,
        stream: Any,
        tracker: "CostTracker",
        default_tags: Dict[str, str],
        tags: Optional[Dict[str, str]],
        model_hint: Optional[str],
        start_time: float,
    ):
        self._stream = stream
        self._tracker = tracker
        self._default_tags = default_tags
        self._tags = tags
        self._model_hint = model_hint
        self._start_time = start_time
        self._input_tokens = 0
        self._output_tokens = 0
        self._model = "unknown"

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._stream.__exit__(*args)

        # Record the call
        latency_ms = int((time.time() - self._start_time) * 1000)

        all_tags = dict(self._default_tags)
        if self._tags:
            all_tags.update(self._tags)

        model = self._model if self._model != "unknown" else (self._model_hint or "unknown")

        self._tracker.record(
            provider="anthropic",
            model=model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            tags=all_tags,
            success=True,
            latency_ms=latency_ms,
        )

    def __iter__(self):
        for event in self._stream:
            # Track usage from events
            self._handle_event(event)
            yield event

    def _handle_event(self, event: Any) -> None:
        """Extract usage info from streaming event."""
        event_type = getattr(event, "type", "")

        if event_type == "message_start":
            if hasattr(event, "message"):
                self._model = getattr(event.message, "model", self._model)
                if hasattr(event.message, "usage"):
                    self._input_tokens = getattr(event.message.usage, "input_tokens", 0)

        elif event_type == "message_delta":
            if hasattr(event, "usage"):
                self._output_tokens = getattr(event.usage, "output_tokens", 0)
