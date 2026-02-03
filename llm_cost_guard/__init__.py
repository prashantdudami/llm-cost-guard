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
from llm_cost_guard.audit import (
    AuditLogger,
    AuditBackend,
    AuditEvent,
    AuditEventType,
    LoggingAuditBackend,
    FileAuditBackend,
)
from llm_cost_guard.encryption import (
    EncryptionProvider,
    NoEncryption,
    FernetEncryption,
    FieldEncryption,
    get_encryption_provider,
)
from llm_cost_guard.secrets import (
    SecretsProvider,
    EnvironmentSecretsProvider,
    FileSecretsProvider,
    get_secrets_provider,
)
from llm_cost_guard.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    retry_with_backoff,
    RetryConfig,
    ResilientOperation,
)
from llm_cost_guard.metrics import (
    MetricsExporter,
    NoOpExporter,
    LoggingExporter,
    get_metrics_exporter,
)

__version__ = "0.3.1"

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
    # Audit
    "AuditLogger",
    "AuditBackend",
    "AuditEvent",
    "AuditEventType",
    "LoggingAuditBackend",
    "FileAuditBackend",
    # Encryption
    "EncryptionProvider",
    "NoEncryption",
    "FernetEncryption",
    "FieldEncryption",
    "get_encryption_provider",
    # Secrets
    "SecretsProvider",
    "EnvironmentSecretsProvider",
    "FileSecretsProvider",
    "get_secrets_provider",
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    "retry_with_backoff",
    "RetryConfig",
    "ResilientOperation",
    # Metrics
    "MetricsExporter",
    "NoOpExporter",
    "LoggingExporter",
    "get_metrics_exporter",
    # Exceptions
    "LLMCostGuardError",
    "BudgetExceededError",
    "PricingNotFoundError",
    "TokenCountError",
    "TrackingUnavailableError",
    "RateLimitExceededError",
]
