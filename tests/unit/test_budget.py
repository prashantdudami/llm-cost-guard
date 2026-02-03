"""
Unit tests for budget enforcement.
"""

import pytest
from datetime import datetime, timedelta

from llm_cost_guard.budget import Budget, BudgetAction, BudgetTracker


class TestBudget:
    """Tests for Budget dataclass."""

    def test_budget_creation(self):
        """Test budget creation."""
        budget = Budget(
            name="daily",
            limit=10.00,
            period="day",
            action=BudgetAction.WARN,
        )

        assert budget.name == "daily"
        assert budget.limit == 10.00
        assert budget.period == "day"
        assert budget.action == BudgetAction.WARN

    def test_budget_with_tags(self):
        """Test budget with tag filtering."""
        budget = Budget(
            name="team-budget",
            limit=50.00,
            period="day",
            action=BudgetAction.BLOCK,
            tags={"team": "search"},
        )

        assert budget.matches_tags({"team": "search"})
        assert budget.matches_tags({"team": "search", "feature": "autocomplete"})
        assert not budget.matches_tags({"team": "chat"})
        assert not budget.matches_tags({})

    def test_budget_no_tags(self):
        """Test budget without tags matches everything."""
        budget = Budget(
            name="global",
            limit=100.00,
            period="day",
            action=BudgetAction.WARN,
        )

        assert budget.matches_tags({})
        assert budget.matches_tags({"team": "any"})


class TestBudgetTracker:
    """Tests for BudgetTracker class."""

    def test_tracker_creation(self):
        """Test budget tracker creation."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.WARN)
        ])

        assert tracker.get_budget("daily") is not None
        assert tracker.get_budget("nonexistent") is None

    def test_record_cost(self):
        """Test recording costs."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.WARN)
        ])

        tracker.record_cost(1.00)
        assert tracker.get_spending("daily") == 1.00

        tracker.record_cost(2.50)
        assert tracker.get_spending("daily") == 3.50

    def test_check_budget_not_exceeded(self):
        """Test check_budget when not exceeded."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.BLOCK)
        ])

        exceeded = tracker.check_budget(5.00)
        assert len(exceeded) == 0

    def test_check_budget_exceeded(self):
        """Test check_budget when exceeded."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.BLOCK)
        ])

        tracker.record_cost(8.00)
        exceeded = tracker.check_budget(5.00)  # Would exceed limit

        assert len(exceeded) == 1
        assert exceeded[0][0].name == "daily"
        assert exceeded[0][1] == BudgetAction.BLOCK

    def test_get_remaining(self):
        """Test get_remaining budget."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.WARN)
        ])

        assert tracker.get_remaining("daily") == 10.00

        tracker.record_cost(3.00)
        assert tracker.get_remaining("daily") == 7.00

    def test_get_utilization(self):
        """Test budget utilization percentage."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.WARN)
        ])

        assert tracker.get_utilization("daily") == 0.0

        tracker.record_cost(5.00)
        assert tracker.get_utilization("daily") == 50.0

        tracker.record_cost(5.00)
        assert tracker.get_utilization("daily") == 100.0

    def test_warning_threshold(self):
        """Test warning threshold triggers."""
        warnings = []

        tracker = BudgetTracker([
            Budget(
                name="daily",
                limit=10.00,
                period="day",
                action=BudgetAction.WARN,
                warning_threshold=0.8,
            )
        ])

        tracker.on_warning(lambda budget, current: warnings.append((budget.name, current)))

        # Below threshold - no warning
        tracker.record_cost(5.00)
        assert len(warnings) == 0

        # At threshold - warning
        tracker.record_cost(4.00)  # Total: 9.00 (90%)
        assert len(warnings) == 1
        assert warnings[0][0] == "daily"

    def test_exceeded_callback(self):
        """Test exceeded callback triggers."""
        exceeded_events = []

        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.BLOCK)
        ])

        tracker.on_exceeded(lambda budget: exceeded_events.append(budget.name))

        tracker.record_cost(12.00)  # Exceeds limit
        assert len(exceeded_events) == 1
        assert exceeded_events[0] == "daily"

    def test_reset_budget(self):
        """Test resetting budget."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.WARN)
        ])

        tracker.record_cost(5.00)
        assert tracker.get_spending("daily") == 5.00

        tracker.reset("daily")
        assert tracker.get_spending("daily") == 0.00

    def test_add_remove_budget(self):
        """Test adding and removing budgets."""
        tracker = BudgetTracker()

        assert tracker.get_budget("test") is None

        tracker.add_budget(Budget(name="test", limit=10.00, period="day", action=BudgetAction.WARN))
        assert tracker.get_budget("test") is not None

        tracker.remove_budget("test")
        assert tracker.get_budget("test") is None

    def test_per_request_budget(self):
        """Test per-request budget resets each call."""
        tracker = BudgetTracker([
            Budget(name="per-request", limit=1.00, period="request", action=BudgetAction.BLOCK)
        ])

        # First request
        exceeded = tracker.check_budget(0.50)
        assert len(exceeded) == 0

        # Reset happens automatically for per-request
        exceeded = tracker.check_budget(0.50)
        assert len(exceeded) == 0

    def test_reservation(self):
        """Test budget reservation."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.BLOCK)
        ])

        # Reserve budget
        reservation_id = tracker.reserve(5.00)
        assert reservation_id is not None
        assert tracker.get_spending("daily") == 5.00

        # Finalize with lower actual cost
        tracker.finalize(reservation_id, 3.00, 5.00)
        assert tracker.get_spending("daily") == 3.00

    def test_reservation_release(self):
        """Test releasing a reservation."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.BLOCK)
        ])

        reservation_id = tracker.reserve(5.00)
        assert tracker.get_spending("daily") == 5.00

        tracker.release(reservation_id, 5.00)
        assert tracker.get_spending("daily") == 0.00

    def test_reservation_blocked(self):
        """Test reservation when budget would be exceeded."""
        tracker = BudgetTracker([
            Budget(name="daily", limit=10.00, period="day", action=BudgetAction.BLOCK)
        ])

        tracker.record_cost(8.00)

        # Try to reserve more than remaining
        reservation_id = tracker.reserve(5.00)
        assert reservation_id is None
