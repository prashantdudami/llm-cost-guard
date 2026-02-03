"""
Custom exceptions for LLM Cost Guard.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from llm_cost_guard.budget import Budget


class LLMCostGuardError(Exception):
    """Base exception for LLM Cost Guard."""

    pass


class BudgetExceededError(LLMCostGuardError):
    """Raised when a budget limit is exceeded."""

    def __init__(
        self,
        message: str,
        budget: Optional["Budget"] = None,
        current: float = 0.0,
        limit: float = 0.0,
    ):
        super().__init__(message)
        self.budget = budget
        self.current = current
        self.limit = limit


class PricingNotFoundError(LLMCostGuardError):
    """Raised when pricing information for a model is not found."""

    def __init__(self, message: str, provider: str = "", model: str = ""):
        super().__init__(message)
        self.provider = provider
        self.model = model


class TokenCountError(LLMCostGuardError):
    """Raised when token counting fails."""

    pass


class TrackingUnavailableError(LLMCostGuardError):
    """Raised when the tracking backend is unavailable."""

    def __init__(self, message: str, backend: str = ""):
        super().__init__(message)
        self.backend = backend


class RateLimitExceededError(LLMCostGuardError):
    """Raised when a rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit_name: str = "",
        current: int = 0,
        limit: int = 0,
        retry_after_seconds: Optional[float] = None,
    ):
        super().__init__(message)
        self.limit_name = limit_name
        self.current = current
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds
