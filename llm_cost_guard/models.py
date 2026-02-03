"""
Data models for LLM Cost Guard.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class ModelType(str, Enum):
    """Types of LLM models."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    COMPLETION = "completion"


@dataclass
class CostRecord:
    """Single LLM call record."""

    timestamp: datetime
    provider: str
    model: str
    model_type: ModelType = ModelType.CHAT
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    latency_ms: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_type: Optional[str] = None
    cached: bool = False
    cache_savings: float = 0.0
    span_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Calculate total cost if not provided."""
        if self.total_cost == 0.0 and (self.input_cost > 0 or self.output_cost > 0):
            self.total_cost = self.input_cost + self.output_cost


@dataclass
class CostReport:
    """Aggregated cost report."""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    cache_hits: int = 0
    cache_savings: float = 0.0
    effective_cost: float = 0.0  # total_cost - cache_savings
    records: List[CostRecord] = field(default_factory=list)
    grouped_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate effective cost."""
        if self.effective_cost == 0.0:
            self.effective_cost = self.total_cost - self.cache_savings


@dataclass
class HealthStatus:
    """Health check status for the tracker."""

    healthy: bool = True
    backend_connected: bool = True
    pricing_fresh: bool = True
    last_record_time: Optional[datetime] = None
    pending_records: int = 0
    errors: List[str] = field(default_factory=list)
    pricing_version: Optional[str] = None
    pricing_last_updated: Optional[datetime] = None


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_cost_per_1k: float
    output_cost_per_1k: float
    cached_input_cost_per_1k: Optional[float] = None
    context_window: int = 128000
    model_type: ModelType = ModelType.CHAT

    # For image models
    image_cost_per_image: Optional[float] = None

    # For audio models
    audio_cost_per_minute: Optional[float] = None

    # For embedding models
    embedding_dimensions: Optional[int] = None


@dataclass
class UsageData:
    """Token usage data from an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    # For non-text models
    image_count: int = 0
    audio_duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
