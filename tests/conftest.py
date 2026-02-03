"""
Pytest configuration and fixtures for LLM Cost Guard tests.
"""

import pytest
from datetime import datetime

from llm_cost_guard import CostTracker, Budget, BudgetAction, RateLimit
from llm_cost_guard.backends.memory import MemoryBackend
from llm_cost_guard.models import CostRecord, ModelType


@pytest.fixture
def memory_backend():
    """Create a clean memory backend."""
    return MemoryBackend()


@pytest.fixture
def tracker():
    """Create a basic CostTracker with memory backend."""
    return CostTracker(backend="memory")


@pytest.fixture
def tracker_with_budget():
    """Create a CostTracker with budget configured."""
    return CostTracker(
        backend="memory",
        budgets=[
            Budget(
                name="daily",
                limit=10.00,
                period="day",
                action=BudgetAction.WARN,
            ),
            Budget(
                name="per-request",
                limit=0.50,
                period="request",
                action=BudgetAction.BLOCK,
            ),
        ],
    )


@pytest.fixture
def tracker_with_rate_limits():
    """Create a CostTracker with rate limits configured."""
    return CostTracker(
        backend="memory",
        rate_limits=[
            RateLimit(
                name="requests-per-minute",
                limit=10,
                period="minute",
                scope="global",
            ),
        ],
    )


@pytest.fixture
def sample_cost_record():
    """Create a sample CostRecord."""
    return CostRecord(
        timestamp=datetime.now(),
        provider="openai",
        model="gpt-4o",
        model_type=ModelType.CHAT,
        input_tokens=100,
        output_tokens=50,
        input_cost=0.00025,
        output_cost=0.0005,
        total_cost=0.00075,
        latency_ms=500,
        tags={"team": "search", "feature": "autocomplete"},
        metadata={},
        success=True,
        error_type=None,
        cached=False,
        cache_savings=0.0,
        span_id=None,
    )


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""

    class MockUsage:
        prompt_tokens = 100
        completion_tokens = 50
        total_tokens = 150
        prompt_tokens_details = None

    class MockResponse:
        model = "gpt-4o"
        usage = MockUsage()

    return MockResponse()


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic response."""

    class MockUsage:
        input_tokens = 100
        output_tokens = 50

    class MockResponse:
        model = "claude-3-5-sonnet-20241022"
        usage = MockUsage()

    return MockResponse()
