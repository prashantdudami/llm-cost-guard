#!/usr/bin/env python3
"""
LLM Cost Guard - Budget Alerts Example

This example demonstrates how to set up budget enforcement with
custom alert handlers for warnings and exceeded limits.
"""

from llm_cost_guard import CostTracker, Budget, BudgetAction
from llm_cost_guard.exceptions import BudgetExceededError


def main():
    print("=== LLM Cost Guard - Budget Alerts Example ===\n")

    # ==========================================================================
    # Configure Budgets
    # ==========================================================================

    tracker = CostTracker(
        backend="memory",
        budgets=[
            # Global daily budget
            Budget(
                name="daily-global",
                limit=100.00,
                period="day",
                action=BudgetAction.WARN,
                warning_threshold=0.8,  # Warn at 80%
            ),
            # Per-request limit to prevent runaway costs
            Budget(
                name="per-request",
                limit=1.00,  # $1 max per request
                period="request",
                action=BudgetAction.BLOCK,
            ),
            # Team-specific budget
            Budget(
                name="team-search-daily",
                limit=25.00,
                period="day",
                action=BudgetAction.WARN,
                tags={"team": "search"},  # Only applies to search team
                warning_threshold=0.7,
            ),
            # Hourly burst protection
            Budget(
                name="hourly-burst",
                limit=10.00,
                period="hour",
                action=BudgetAction.BLOCK,
            ),
        ],
    )

    # ==========================================================================
    # Set Up Alert Handlers
    # ==========================================================================

    warning_log = []
    exceeded_log = []

    @tracker.on_budget_warning
    def handle_budget_warning(budget: Budget, current: float):
        """
        Called when spending reaches the warning threshold.

        In production, you might:
        - Send a Slack message
        - Write to a monitoring system
        - Send an email alert
        """
        percentage = (current / budget.limit) * 100
        message = f"⚠️  BUDGET WARNING: '{budget.name}' at {percentage:.1f}% (${current:.2f}/${budget.limit:.2f})"
        warning_log.append(message)
        print(message)

    @tracker.on_budget_exceeded
    def handle_budget_exceeded(budget: Budget):
        """
        Called when a budget is exceeded.

        In production, you might:
        - Page on-call
        - Trigger circuit breaker
        - Disable non-critical features
        """
        message = f"🚨 BUDGET EXCEEDED: '{budget.name}' - Limit: ${budget.limit:.2f}"
        exceeded_log.append(message)
        print(message)

    # ==========================================================================
    # Simulate API Calls
    # ==========================================================================

    print("Simulating API calls with various costs...\n")

    # Regular calls for search team
    print("--- Search Team Calls ---")
    for i in range(5):
        try:
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=1000 * (i + 1),
                output_tokens=500 * (i + 1),
                tags={"team": "search", "feature": "autocomplete"},
            )
            print(f"  Search call {i + 1}: OK")
        except BudgetExceededError as e:
            print(f"  Search call {i + 1}: BLOCKED - {e}")

    # Regular calls for chat team
    print("\n--- Chat Team Calls ---")
    for i in range(3):
        try:
            tracker.record(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                input_tokens=2000,
                output_tokens=1000,
                tags={"team": "chat", "feature": "support"},
            )
            print(f"  Chat call {i + 1}: OK")
        except BudgetExceededError as e:
            print(f"  Chat call {i + 1}: BLOCKED - {e}")

    # Try a large call that might exceed per-request limit
    print("\n--- Large Request Test ---")
    try:
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100000,  # Large context
            output_tokens=50000,
            tags={"team": "analytics", "feature": "report"},
        )
        print("  Large call: OK")
    except BudgetExceededError as e:
        print(f"  Large call: BLOCKED - Per-request limit exceeded")

    # ==========================================================================
    # Budget Status Report
    # ==========================================================================

    print("\n" + "=" * 60)
    print("BUDGET STATUS REPORT")
    print("=" * 60)

    budgets = ["daily-global", "per-request", "team-search-daily", "hourly-burst"]

    for budget_name in budgets:
        budget = tracker.get_budget(budget_name)
        if budget:
            utilization = tracker.get_budget_utilization(budget_name)
            remaining = budget.limit * (1 - utilization / 100)
            status = "⚠️  WARNING" if utilization >= budget.warning_threshold * 100 else "✅ OK"
            print(f"\n{budget_name}:")
            print(f"  Limit: ${budget.limit:.2f}")
            print(f"  Utilization: {utilization:.1f}%")
            print(f"  Remaining: ${remaining:.2f}")
            print(f"  Status: {status}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 60)
    print("ALERT SUMMARY")
    print("=" * 60)

    if warning_log:
        print(f"\nWarnings triggered: {len(warning_log)}")
        for msg in warning_log:
            print(f"  {msg}")
    else:
        print("\nNo warnings triggered")

    if exceeded_log:
        print(f"\nExceeded events: {len(exceeded_log)}")
        for msg in exceeded_log:
            print(f"  {msg}")
    else:
        print("\nNo budgets exceeded")

    # Get overall report
    report = tracker.daily_report()
    print(f"\nTotal API calls: {report.total_calls}")
    print(f"Total cost: ${report.total_cost:.2f}")

    tracker.close()


if __name__ == "__main__":
    main()
