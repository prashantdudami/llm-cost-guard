#!/usr/bin/env python3
"""
LLM Cost Guard - Quick Start Example

This example demonstrates the basic usage of LLM Cost Guard for tracking
LLM API costs.
"""

from llm_cost_guard import CostTracker, Budget, BudgetAction

# =============================================================================
# Basic Usage
# =============================================================================

# Create a cost tracker with default settings (in-memory storage)
tracker = CostTracker()

# Manually record an API call (when you have the token counts)
record = tracker.record(
    provider="openai",
    model="gpt-4o",
    input_tokens=1500,
    output_tokens=500,
    tags={"feature": "summarization", "user": "demo"},
)

print("=== Basic Recording ===")
print(f"Model: {record.model}")
print(f"Input tokens: {record.input_tokens}")
print(f"Output tokens: {record.output_tokens}")
print(f"Input cost: ${record.input_cost:.6f}")
print(f"Output cost: ${record.output_cost:.6f}")
print(f"Total cost: ${record.total_cost:.6f}")

# =============================================================================
# Using the Decorator
# =============================================================================

print("\n=== Using Decorator ===")

# You can use the @tracker.track decorator to automatically track calls
# This works with any function that returns an OpenAI/Anthropic-style response


@tracker.track(tags={"feature": "chat"})
def simulated_openai_call():
    """Simulate an OpenAI API call by returning a mock response."""

    class MockUsage:
        prompt_tokens = 250
        completion_tokens = 100
        total_tokens = 350
        prompt_tokens_details = None

    class MockResponse:
        model = "gpt-4o-mini"
        usage = MockUsage()

    return MockResponse()


# Call the decorated function
response = simulated_openai_call()
print(f"Simulated call completed")

# Check the last recorded call
last = tracker.last_call()
if last:
    print(f"Last call model: {last.model}")
    print(f"Last call cost: ${last.total_cost:.6f}")

# =============================================================================
# With Budget Enforcement
# =============================================================================

print("\n=== Budget Enforcement ===")

# Create a tracker with budget limits
budget_tracker = CostTracker(
    backend="memory",
    budgets=[
        Budget(
            name="daily-limit",
            limit=5.00,  # $5 daily limit
            period="day",
            action=BudgetAction.WARN,  # Just warn, don't block
            warning_threshold=0.5,  # Warn at 50%
        ),
        Budget(
            name="per-request-limit",
            limit=0.10,  # $0.10 per request
            period="request",
            action=BudgetAction.BLOCK,  # Block if exceeded
        ),
    ],
)


# Set up warning handler
@budget_tracker.on_budget_warning
def handle_warning(budget, current):
    print(f"  ⚠️  Budget warning: '{budget.name}' at ${current:.2f}/${budget.limit:.2f}")


# Make some tracked calls
for i in range(3):
    try:
        budget_tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=500,
            output_tokens=200,
            tags={"iteration": str(i)},
        )
        print(f"  Call {i + 1}: Recorded successfully")
    except Exception as e:
        print(f"  Call {i + 1}: {type(e).__name__}: {e}")

# Check budget utilization
utilization = budget_tracker.get_budget_utilization("daily-limit")
print(f"\nDaily budget utilization: {utilization:.1f}%")

# =============================================================================
# Reporting
# =============================================================================

print("\n=== Cost Report ===")

# Get a daily report from the original tracker
report = tracker.daily_report()
print(f"Total calls today: {report.total_calls}")
print(f"Total cost today: ${report.total_cost:.6f}")
print(f"Total input tokens: {report.total_input_tokens:,}")
print(f"Total output tokens: {report.total_output_tokens:,}")

# =============================================================================
# Health Check
# =============================================================================

print("\n=== Health Check ===")

health = tracker.health_check()
print(f"Tracker healthy: {health.healthy}")
print(f"Backend connected: {health.backend_connected}")
print(f"Pricing data fresh: {health.pricing_fresh}")

# Clean up
tracker.close()
budget_tracker.close()

print("\n✅ Quick start example completed!")
