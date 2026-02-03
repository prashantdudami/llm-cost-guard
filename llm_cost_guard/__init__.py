"""
LLM Cost Guard - Real-time cost tracking, budget enforcement, and usage analytics for LLM applications.
"""

from llm_cost_guard.tracker import CostTracker
from llm_cost_guard.budget import Budget, BudgetAction
from llm_cost_guard.rate_limit import RateLimit
from llm_cost_guard.span import Span
from llm_cost_guard.models import CostRecord, CostReport, HealthStatus
from llm_cost_guard.exceptions import (
    LLMCostGuardError,
    BudgetExceededError,
    PricingNotFoundError,
    TokenCountError,
    TrackingUnavailableError,
    RateLimitExceededError,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "CostTracker",
    "Budget",
    "BudgetAction",
    "RateLimit",
    "Span",
    # Models
    "CostRecord",
    "CostReport",
    "HealthStatus",
    # Exceptions
    "LLMCostGuardError",
    "BudgetExceededError",
    "PricingNotFoundError",
    "TokenCountError",
    "TrackingUnavailableError",
    "RateLimitExceededError",
]
