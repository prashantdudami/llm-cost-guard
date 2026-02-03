"""
Base backend interface for LLM Cost Guard.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from llm_cost_guard.models import CostRecord, CostReport


class Backend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_record(self, record: CostRecord) -> None:
        """Save a cost record."""
        pass

    @abstractmethod
    def save_records(self, records: List[CostRecord]) -> None:
        """Save multiple cost records."""
        pass

    @abstractmethod
    def get_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[CostRecord]:
        """
        Retrieve cost records with optional filters.

        Args:
            start_date: Filter records after this date
            end_date: Filter records before this date
            tags: Filter by tag key-value pairs
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of matching CostRecord objects
        """
        pass

    @abstractmethod
    def get_total_cost(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get total cost for the given filters."""
        pass

    @abstractmethod
    def get_aggregated_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        group_by: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregated costs grouped by specified fields.

        Args:
            start_date: Filter records after this date
            end_date: Filter records before this date
            tags: Filter by tag key-value pairs
            group_by: Fields to group by (e.g., ["provider", "model", "team"])

        Returns:
            Dictionary with grouped cost data
        """
        pass

    @abstractmethod
    def get_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        group_by: Optional[List[str]] = None,
    ) -> CostReport:
        """Generate a cost report."""
        pass

    @abstractmethod
    def delete_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Delete records matching the filters.

        Returns:
            Number of records deleted
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is healthy and connected."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the backend connection."""
        pass

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
