"""
Wrapped OpenAI client with automatic cost tracking.
"""

import time
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_cost_guard import CostTracker


class TrackedOpenAI:
    """
    OpenAI client wrapper with automatic cost tracking.

    Usage:
        from llm_cost_guard import CostTracker
        from llm_cost_guard.clients import TrackedOpenAI

        tracker = CostTracker()
        client = TrackedOpenAI(tracker=tracker)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Cost is automatically tracked
    """

    def __init__(
        self,
        tracker: "CostTracker",
        client: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
        **openai_kwargs: Any,
    ):
        """
        Initialize the tracked OpenAI client.

        Args:
            tracker: CostTracker instance
            client: Optional existing OpenAI client to wrap
            tags: Default tags for all calls
            **openai_kwargs: Arguments to pass to OpenAI client
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI is required for this client. Install with: pip install openai"
            )

        self._tracker = tracker
        self._default_tags = tags or {}
        self._client = client or OpenAI(**openai_kwargs)

        # Create wrapped interface
        self.chat = _TrackedChat(self._client.chat, self._tracker, self._default_tags)
        self.completions = _TrackedCompletions(
            self._client.completions, self._tracker, self._default_tags
        )
        self.embeddings = _TrackedEmbeddings(
            self._client.embeddings, self._tracker, self._default_tags
        )

    @property
    def models(self):
        """Access the models API."""
        return self._client.models

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _TrackedChat:
    """Wrapped chat completions API."""

    def __init__(self, chat, tracker: "CostTracker", default_tags: Dict[str, str]):
        self._chat = chat
        self._tracker = tracker
        self._default_tags = default_tags
        self.completions = _TrackedChatCompletions(
            chat.completions, tracker, default_tags
        )


class _TrackedChatCompletions:
    """Wrapped chat.completions API."""

    def __init__(self, completions, tracker: "CostTracker", default_tags: Dict[str, str]):
        self._completions = completions
        self._tracker = tracker
        self._default_tags = default_tags

    def create(
        self,
        *,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion with tracking."""
        start_time = time.time()
        success = True
        error_type = None
        response = None

        try:
            response = self._completions.create(**kwargs)
            return response
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            if response is not None:
                self._record_response(response, tags, success, error_type, latency_ms)

    def _record_response(
        self,
        response: Any,
        tags: Optional[Dict[str, str]],
        success: bool,
        error_type: Optional[str],
        latency_ms: int,
    ) -> None:
        """Record the response with the tracker."""
        from llm_cost_guard.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        usage = provider.extract_usage(response)
        model = provider.extract_model(response)

        all_tags = dict(self._default_tags)
        if tags:
            all_tags.update(tags)

        self._tracker.record(
            provider="openai",
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tags=all_tags,
            success=success,
            error_type=error_type,
            latency_ms=latency_ms,
            cached_tokens=usage.cached_tokens,
        )


class _TrackedCompletions:
    """Wrapped completions API (legacy)."""

    def __init__(self, completions, tracker: "CostTracker", default_tags: Dict[str, str]):
        self._completions = completions
        self._tracker = tracker
        self._default_tags = default_tags

    def create(
        self,
        *,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a completion with tracking."""
        start_time = time.time()
        success = True
        error_type = None
        response = None

        try:
            response = self._completions.create(**kwargs)
            return response
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            if response is not None:
                from llm_cost_guard.providers.openai import OpenAIProvider

                provider = OpenAIProvider()
                usage = provider.extract_usage(response)
                model = provider.extract_model(response)

                all_tags = dict(self._default_tags)
                if tags:
                    all_tags.update(tags)

                self._tracker.record(
                    provider="openai",
                    model=model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    tags=all_tags,
                    success=success,
                    error_type=error_type,
                    latency_ms=latency_ms,
                )


class _TrackedEmbeddings:
    """Wrapped embeddings API."""

    def __init__(self, embeddings, tracker: "CostTracker", default_tags: Dict[str, str]):
        self._embeddings = embeddings
        self._tracker = tracker
        self._default_tags = default_tags

    def create(
        self,
        *,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create embeddings with tracking."""
        start_time = time.time()
        success = True
        error_type = None
        response = None

        try:
            response = self._embeddings.create(**kwargs)
            return response
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            if response is not None:
                from llm_cost_guard.providers.openai import OpenAIProvider

                provider = OpenAIProvider()
                usage = provider.extract_usage(response)
                model = kwargs.get("model", "unknown")

                all_tags = dict(self._default_tags)
                if tags:
                    all_tags.update(tags)

                self._tracker.record(
                    provider="openai",
                    model=model,
                    input_tokens=usage.input_tokens,
                    output_tokens=0,  # Embeddings don't have output tokens
                    tags=all_tags,
                    success=success,
                    error_type=error_type,
                    latency_ms=latency_ms,
                )
