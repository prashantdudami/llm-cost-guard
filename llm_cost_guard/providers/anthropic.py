"""
Anthropic provider for LLM Cost Guard.
"""

from typing import Any

from llm_cost_guard.models import UsageData
from llm_cost_guard.providers.base import Provider


class AnthropicProvider(Provider):
    """Anthropic API provider."""

    @property
    def name(self) -> str:
        return "anthropic"

    def extract_usage(self, response: Any) -> UsageData:
        """Extract token usage from an Anthropic API response."""
        usage = UsageData()

        # Handle dictionary response
        if isinstance(response, dict):
            usage_data = response.get("usage", {})
            usage.input_tokens = usage_data.get("input_tokens", 0)
            usage.output_tokens = usage_data.get("output_tokens", 0)

            # Check for cached tokens
            usage.cached_tokens = usage_data.get("cache_read_input_tokens", 0)
            cache_creation = usage_data.get("cache_creation_input_tokens", 0)

            # Total tokens
            usage.total_tokens = usage.input_tokens + usage.output_tokens

            return usage

        # Handle Anthropic client response object
        if hasattr(response, "usage") and response.usage is not None:
            usage.input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            usage.output_tokens = getattr(response.usage, "output_tokens", 0) or 0

            # Anthropic prompt caching
            if hasattr(response.usage, "cache_read_input_tokens"):
                usage.cached_tokens = response.usage.cache_read_input_tokens or 0

            usage.total_tokens = usage.input_tokens + usage.output_tokens

        return usage

    def extract_model(self, response: Any) -> str:
        """Extract the model name from an Anthropic API response."""
        if isinstance(response, dict):
            return response.get("model", "unknown")

        if hasattr(response, "model"):
            return response.model or "unknown"

        return "unknown"

    def extract_cached_tokens(self, response: Any) -> int:
        """Extract cached token count from an Anthropic API response."""
        usage = self.extract_usage(response)
        return usage.cached_tokens

    def normalize_model_name(self, model: str) -> str:
        """Normalize Anthropic model name."""
        # Anthropic model names are usually already normalized
        return model


class AnthropicStreamingHandler:
    """Handler for streaming Anthropic responses."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.model = "unknown"
        self._started = False

    def handle_event(self, event: Any) -> None:
        """Process a streaming event."""
        if isinstance(event, dict):
            event_type = event.get("type", "")

            if event_type == "message_start":
                message = event.get("message", {})
                self.model = message.get("model", self.model)
                usage = message.get("usage", {})
                self.input_tokens = usage.get("input_tokens", 0)

            elif event_type == "message_delta":
                usage = event.get("usage", {})
                self.output_tokens = usage.get("output_tokens", 0)

        else:
            # Handle event objects
            event_type = getattr(event, "type", "")

            if event_type == "message_start":
                if hasattr(event, "message"):
                    self.model = getattr(event.message, "model", self.model)
                    if hasattr(event.message, "usage"):
                        self.input_tokens = getattr(event.message.usage, "input_tokens", 0)

            elif event_type == "message_delta":
                if hasattr(event, "usage"):
                    self.output_tokens = getattr(event.usage, "output_tokens", 0)

    def get_usage(self) -> UsageData:
        """Get final usage data."""
        return UsageData(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )
