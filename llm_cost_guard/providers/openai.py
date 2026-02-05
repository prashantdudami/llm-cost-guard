"""
OpenAI provider for LLM Cost Guard.
"""

from typing import Any

from llm_cost_guard.models import UsageData
from llm_cost_guard.providers.base import Provider


class OpenAIProvider(Provider):
    """OpenAI API provider."""

    @property
    def name(self) -> str:
        return "openai"

    def extract_usage(self, response: Any) -> UsageData:
        """Extract token usage from an OpenAI API response."""
        usage = UsageData()

        # Handle dictionary response
        if isinstance(response, dict):
            usage_data = response.get("usage", {})
            usage.input_tokens = usage_data.get("prompt_tokens", 0)
            usage.output_tokens = usage_data.get("completion_tokens", 0)
            usage.total_tokens = usage_data.get("total_tokens", 0)

            # Check for cached tokens (prompt caching)
            prompt_tokens_details = usage_data.get("prompt_tokens_details", {})
            if prompt_tokens_details:
                usage.cached_tokens = prompt_tokens_details.get("cached_tokens", 0)

            return usage

        # Handle OpenAI client response object
        if hasattr(response, "usage") and response.usage is not None:
            usage.input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            usage.output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            usage.total_tokens = getattr(response.usage, "total_tokens", 0) or 0

            # Check for prompt caching details
            if hasattr(response.usage, "prompt_tokens_details"):
                details = response.usage.prompt_tokens_details
                if details and hasattr(details, "cached_tokens"):
                    usage.cached_tokens = details.cached_tokens or 0

        return usage

    def extract_model(self, response: Any) -> str:
        """Extract the model name from an OpenAI API response."""
        if isinstance(response, dict):
            return response.get("model", "unknown")

        if hasattr(response, "model"):
            return response.model or "unknown"

        return "unknown"

    def extract_cached_tokens(self, response: Any) -> int:
        """Extract cached token count from an OpenAI API response."""
        usage = self.extract_usage(response)
        return usage.cached_tokens

    def normalize_model_name(self, model: str) -> str:
        """Normalize OpenAI model name."""
        # Remove date suffixes for pricing lookup
        # e.g., "gpt-4-0613" -> "gpt-4"
        parts = model.split("-")
        if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) >= 4:
            return "-".join(parts[:-1])
        return model


class OpenAIStreamingHandler:
    """Handler for streaming OpenAI responses."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.model = "unknown"
        self._chunk_count = 0

    def handle_chunk(self, chunk: Any) -> None:
        """Process a streaming chunk."""
        self._chunk_count += 1

        # Extract model from first chunk
        if self._chunk_count == 1:
            if isinstance(chunk, dict):
                self.model = chunk.get("model", self.model)
            elif hasattr(chunk, "model"):
                self.model = chunk.model or self.model

        # Some providers include usage in the final chunk
        if isinstance(chunk, dict):
            if "usage" in chunk and chunk["usage"]:
                self.input_tokens = chunk["usage"].get("prompt_tokens", 0)
                self.output_tokens = chunk["usage"].get("completion_tokens", 0)
        elif hasattr(chunk, "usage") and chunk.usage:
            self.input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
            self.output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

    def get_usage(self) -> UsageData:
        """Get final usage data."""
        return UsageData(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )
