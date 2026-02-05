"""
Base provider interface for LLM Cost Guard.
"""

from abc import ABC, abstractmethod
from typing import Any

from llm_cost_guard.models import UsageData


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name."""
        pass

    @abstractmethod
    def extract_usage(self, response: Any) -> UsageData:
        """
        Extract token usage from an API response.

        Args:
            response: The raw API response object

        Returns:
            UsageData with token counts
        """
        pass

    @abstractmethod
    def extract_model(self, response: Any) -> str:
        """
        Extract the model name from an API response.

        Args:
            response: The raw API response object

        Returns:
            Model name string
        """
        pass

    def extract_cached_tokens(self, response: Any) -> int:
        """
        Extract cached token count from an API response.

        Args:
            response: The raw API response object

        Returns:
            Number of cached tokens (0 if not applicable)
        """
        return 0

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming."""
        return True

    def normalize_model_name(self, model: str) -> str:
        """
        Normalize model name for consistent pricing lookup.

        Args:
            model: Raw model name from API

        Returns:
            Normalized model name
        """
        return model
