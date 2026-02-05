"""
LangChain integration for LLM Cost Guard.
"""

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore
    LLMResult = None  # type: ignore


class CostTrackingCallback(BaseCallbackHandler):
    """
    LangChain callback handler for cost tracking.

    Usage:
        from llm_cost_guard import CostTracker
        from llm_cost_guard.integrations.langchain import CostTrackingCallback

        tracker = CostTracker()
        callback = CostTrackingCallback(tracker)

        llm = ChatOpenAI(model="gpt-4o", callbacks=[callback])
        result = llm.invoke("Hello!")
    """

    def __init__(
        self,
        tracker: Any,  # CostTracker
        tags: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the callback handler.

        Args:
            tracker: CostTracker instance
            tags: Default tags to apply to all tracked calls
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this integration. "
                "Install with: pip install llm-cost-guard[langchain]"
            )

        super().__init__()
        self._tracker = tracker
        self._default_tags = tags or {}

        # Track in-flight calls
        self._run_info: dict[str, dict[str, Any]] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of an LLM call."""
        import time

        self._run_info[str(run_id)] = {
            "start_time": time.time(),
            "model": serialized.get("kwargs", {}).get("model_name", "unknown"),
            "prompts": prompts,
            "tags": tags or [],
            "metadata": metadata or {},
        }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of a chat model call."""
        import time

        model = serialized.get("kwargs", {}).get("model_name")
        if not model:
            model = serialized.get("kwargs", {}).get("model", "unknown")

        self._run_info[str(run_id)] = {
            "start_time": time.time(),
            "model": model,
            "messages": messages,
            "tags": tags or [],
            "metadata": metadata or {},
        }

    def on_llm_end(
        self,
        response: "LLMResult",
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Record the end of an LLM call."""
        import time

        run_id_str = str(run_id)
        if run_id_str not in self._run_info:
            return

        run_info = self._run_info.pop(run_id_str)
        latency_ms = int((time.time() - run_info["start_time"]) * 1000)

        # Extract usage from response
        input_tokens = 0
        output_tokens = 0

        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

            # Also check for model-specific usage
            if "usage" in response.llm_output:
                usage = response.llm_output["usage"]
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", input_tokens))
                output_tokens = usage.get(
                    "completion_tokens", usage.get("output_tokens", output_tokens)
                )

        # Detect provider from model
        from llm_cost_guard.providers import detect_provider

        model = run_info["model"]
        provider = detect_provider(model)

        # Build tags
        tags = dict(self._default_tags)
        for tag in run_info.get("tags", []):
            if ":" in tag:
                key, value = tag.split(":", 1)
                tags[key] = value
            else:
                tags[tag] = "true"

        # Record the call
        try:
            self._tracker.record(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tags=tags,
                success=True,
                latency_ms=latency_ms,
                metadata=run_info.get("metadata", {}),
            )
        except Exception as e:
            logger.warning(f"Failed to record LangChain call: {e}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Record an LLM call error."""
        import time

        run_id_str = str(run_id)
        if run_id_str not in self._run_info:
            return

        run_info = self._run_info.pop(run_id_str)
        latency_ms = int((time.time() - run_info["start_time"]) * 1000)

        # Detect provider from model
        from llm_cost_guard.providers import detect_provider

        model = run_info["model"]
        provider = detect_provider(model)

        # Build tags
        tags = dict(self._default_tags)
        for tag in run_info.get("tags", []):
            if ":" in tag:
                key, value = tag.split(":", 1)
                tags[key] = value
            else:
                tags[tag] = "true"

        # Record the failed call
        try:
            self._tracker.record(
                provider=provider,
                model=model,
                input_tokens=0,  # We don't know tokens for failed calls
                output_tokens=0,
                tags=tags,
                success=False,
                error_type=type(error).__name__,
                latency_ms=latency_ms,
                metadata=run_info.get("metadata", {}),
            )
        except Exception as e:
            logger.warning(f"Failed to record LangChain error: {e}")


def track_chain(
    tracker: Any,  # CostTracker
    tags: Optional[dict[str, str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to track costs for an entire LangChain chain.

    Usage:
        @track_chain(tracker, tags={"chain": "rag_pipeline"})
        def my_rag_chain(query):
            # Chain implementation
            return result

    Args:
        tracker: CostTracker instance
        tags: Tags to apply to the tracked span

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracker.span(func.__name__, tags=tags):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
