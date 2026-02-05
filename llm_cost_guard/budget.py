"""
Budget enforcement for LLM Cost Guard.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Literal, Optional

logger = logging.getLogger(__name__)


class BudgetAction(str, Enum):
    """Actions to take when a budget is exceeded."""

    WARN = "warn"
    THROTTLE = "throttle"
    BLOCK = "block"
    CALLBACK = "callback"


BudgetPeriod = Literal["request", "minute", "hour", "day", "week", "month"]


@dataclass
class Budget:
    """Budget configuration."""

    name: str
    limit: float
    period: BudgetPeriod = "day"
    action: BudgetAction = BudgetAction.WARN
    tags: Optional[dict[str, str]] = None
    warning_threshold: float = 0.8  # Warn at 80%
    callback: Optional[Callable[["Budget", float], None]] = None

    def matches_tags(self, call_tags: dict[str, str]) -> bool:
        """Check if this budget applies to the given tags."""
        if self.tags is None:
            return True
        for key, value in self.tags.items():
            if call_tags.get(key) != value:
                return False
        return True


class BudgetTracker:
    """Tracks spending against budgets."""

    def __init__(self, budgets: Optional[list[Budget]] = None):
        self._budgets = budgets or []
        self._spending: dict[str, float] = {}  # budget_name -> current spending
        self._period_start: dict[str, datetime] = {}  # budget_name -> period start
        self._lock = threading.Lock()
        self._warning_callbacks: list[Callable[[Budget, float], None]] = []
        self._exceeded_callbacks: list[Callable[[Budget], None]] = []

        # Initialize budget tracking
        now = datetime.now()
        for budget in self._budgets:
            self._spending[budget.name] = 0.0
            self._period_start[budget.name] = self._get_period_start(budget.period, now)

    def _get_period_start(self, period: BudgetPeriod, now: datetime) -> datetime:
        """Get the start of the current period."""
        if period == "request":
            return now
        elif period == "minute":
            return now.replace(second=0, microsecond=0)
        elif period == "hour":
            return now.replace(minute=0, second=0, microsecond=0)
        elif period == "day":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start - timedelta(days=now.weekday())
        elif period == "month":
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return now

    def _get_period_duration(self, period: BudgetPeriod) -> timedelta:
        """Get the duration of a period."""
        durations = {
            "request": timedelta(seconds=0),
            "minute": timedelta(minutes=1),
            "hour": timedelta(hours=1),
            "day": timedelta(days=1),
            "week": timedelta(weeks=1),
            "month": timedelta(days=30),  # Approximate
        }
        return durations.get(period, timedelta(days=1))

    def _reset_if_new_period(self, budget: Budget, now: datetime) -> None:
        """Reset spending if we've entered a new period."""
        if budget.period == "request":
            self._spending[budget.name] = 0.0
            self._period_start[budget.name] = now
            return

        period_start = self._period_start.get(budget.name)
        if period_start is None:
            self._period_start[budget.name] = self._get_period_start(budget.period, now)
            self._spending[budget.name] = 0.0
            return

        period_duration = self._get_period_duration(budget.period)
        if now >= period_start + period_duration:
            self._spending[budget.name] = 0.0
            self._period_start[budget.name] = self._get_period_start(budget.period, now)

    def add_budget(self, budget: Budget) -> None:
        """Add a new budget."""
        with self._lock:
            self._budgets.append(budget)
            now = datetime.now()
            self._spending[budget.name] = 0.0
            self._period_start[budget.name] = self._get_period_start(budget.period, now)

    def remove_budget(self, name: str) -> bool:
        """Remove a budget by name."""
        with self._lock:
            for i, budget in enumerate(self._budgets):
                if budget.name == name:
                    del self._budgets[i]
                    del self._spending[name]
                    del self._period_start[name]
                    return True
        return False

    def get_budget(self, name: str) -> Optional[Budget]:
        """Get a budget by name."""
        for budget in self._budgets:
            if budget.name == name:
                return budget
        return None

    def get_all_budgets(self) -> list[Budget]:
        """Get all budgets."""
        return list(self._budgets)

    def get_spending(self, budget_name: str) -> float:
        """Get current spending for a budget."""
        with self._lock:
            return self._spending.get(budget_name, 0.0)

    def get_remaining(self, budget_name: str) -> float:
        """Get remaining budget."""
        budget = self.get_budget(budget_name)
        if budget is None:
            return float("inf")
        with self._lock:
            return max(0.0, budget.limit - self._spending.get(budget_name, 0.0))

    def get_utilization(self, budget_name: str) -> float:
        """
        Get budget utilization as a percentage (0-100).

        Returns 0.0 if budget doesn't exist or has zero/negative limit.
        Returns 100.0 cap if spending exceeds limit.
        """
        budget = self.get_budget(budget_name)
        if budget is None or budget.limit <= 0:
            return 0.0
        with self._lock:
            spending = self._spending.get(budget_name, 0.0)
            utilization = (spending / budget.limit) * 100
            # Cap at reasonable maximum
            return min(utilization, 1000.0)  # Allow up to 10x over budget

    def on_warning(self, callback: Callable[[Budget, float], None]) -> None:
        """Register a callback for budget warnings."""
        self._warning_callbacks.append(callback)

    def on_exceeded(self, callback: Callable[[Budget], None]) -> None:
        """Register a callback for budget exceeded events."""
        self._exceeded_callbacks.append(callback)

    def check_budget(
        self, cost: float, tags: Optional[dict[str, str]] = None
    ) -> list[tuple[Budget, BudgetAction]]:
        """
        Check if adding a cost would exceed any budgets.
        Returns list of (budget, action) tuples for budgets that would be exceeded.
        """
        tags = tags or {}
        exceeded = []
        now = datetime.now()

        with self._lock:
            for budget in self._budgets:
                if not budget.matches_tags(tags):
                    continue

                self._reset_if_new_period(budget, now)

                current = self._spending.get(budget.name, 0.0)
                projected = current + cost

                if projected > budget.limit:
                    exceeded.append((budget, budget.action))

        return exceeded

    def record_cost(
        self, cost: float, tags: Optional[dict[str, str]] = None
    ) -> list[tuple[Budget, BudgetAction, float]]:
        """
        Record a cost against matching budgets.
        Returns list of (budget, action, current_spending) for warnings/exceeded.
        """
        tags = tags or {}
        actions = []
        now = datetime.now()

        with self._lock:
            for budget in self._budgets:
                if not budget.matches_tags(tags):
                    continue

                self._reset_if_new_period(budget, now)

                self._spending[budget.name] = self._spending.get(budget.name, 0.0) + cost
                current = self._spending[budget.name]

                # Check for exceeded
                if current > budget.limit:
                    actions.append((budget, budget.action, current))
                    # Trigger exceeded callbacks
                    for callback in self._exceeded_callbacks:
                        try:
                            callback(budget)
                        except Exception as e:
                            logger.error(f"Budget exceeded callback error: {e}")
                # Check for warning
                elif current >= budget.limit * budget.warning_threshold:
                    actions.append((budget, BudgetAction.WARN, current))
                    # Trigger warning callbacks
                    for callback in self._warning_callbacks:
                        try:
                            callback(budget, current)
                        except Exception as e:
                            logger.error(f"Budget warning callback error: {e}")

        return actions

    def reserve(
        self, estimated_cost: float, tags: Optional[dict[str, str]] = None
    ) -> Optional[str]:
        """
        Reserve budget for a call (pessimistic mode).
        Returns a reservation ID if successful, None if budget would be exceeded.
        """
        import uuid

        tags = tags or {}

        exceeded = self.check_budget(estimated_cost, tags)
        blocking = [b for b, action in exceeded if action == BudgetAction.BLOCK]

        if blocking:
            return None

        # Record the reservation
        self.record_cost(estimated_cost, tags)
        return str(uuid.uuid4())

    def finalize(
        self,
        reservation_id: str,
        actual_cost: float,
        estimated_cost: float,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Finalize a reservation with the actual cost.
        Adjusts the difference between estimated and actual.
        """
        tags = tags or {}
        adjustment = actual_cost - estimated_cost

        with self._lock:
            for budget in self._budgets:
                if not budget.matches_tags(tags):
                    continue
                self._spending[budget.name] = self._spending.get(budget.name, 0.0) + adjustment

    def release(
        self, reservation_id: str, estimated_cost: float, tags: Optional[dict[str, str]] = None
    ) -> None:
        """
        Release a reservation (on failure).
        """
        tags = tags or {}

        with self._lock:
            for budget in self._budgets:
                if not budget.matches_tags(tags):
                    continue
                self._spending[budget.name] = max(
                    0.0, self._spending.get(budget.name, 0.0) - estimated_cost
                )

    def reset(self, budget_name: Optional[str] = None) -> None:
        """Reset spending for a specific budget or all budgets."""
        with self._lock:
            if budget_name:
                if budget_name in self._spending:
                    self._spending[budget_name] = 0.0
            else:
                for name in self._spending:
                    self._spending[name] = 0.0
