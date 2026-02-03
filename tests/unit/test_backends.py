"""
Unit tests for storage backends.
"""

import pytest
from datetime import datetime, timedelta

from llm_cost_guard.backends.memory import MemoryBackend
from llm_cost_guard.backends.sqlite import SQLiteBackend
from llm_cost_guard.models import CostRecord, ModelType


class TestMemoryBackend:
    """Tests for MemoryBackend."""

    def test_save_record(self, memory_backend, sample_cost_record):
        """Test saving a single record."""
        memory_backend.save_record(sample_cost_record)
        assert memory_backend.record_count == 1

    def test_save_records(self, memory_backend):
        """Test saving multiple records."""
        records = [
            CostRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                total_cost=0.001,
            )
            for _ in range(5)
        ]

        memory_backend.save_records(records)
        assert memory_backend.record_count == 5

    def test_get_records(self, memory_backend, sample_cost_record):
        """Test retrieving records."""
        memory_backend.save_record(sample_cost_record)

        records = memory_backend.get_records()
        assert len(records) == 1
        assert records[0].provider == "openai"

    def test_get_records_with_date_filter(self, memory_backend):
        """Test filtering records by date."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        old_record = CostRecord(
            timestamp=yesterday,
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            total_cost=0.001,
        )
        new_record = CostRecord(
            timestamp=now,
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            total_cost=0.001,
        )

        memory_backend.save_record(old_record)
        memory_backend.save_record(new_record)

        # Get only today's records
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        records = memory_backend.get_records(start_date=today_start)
        assert len(records) == 1

    def test_get_records_with_tag_filter(self, memory_backend):
        """Test filtering records by tags."""
        record1 = CostRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            total_cost=0.001,
            tags={"team": "search"},
        )
        record2 = CostRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            total_cost=0.001,
            tags={"team": "chat"},
        )

        memory_backend.save_record(record1)
        memory_backend.save_record(record2)

        # Filter by team
        records = memory_backend.get_records(tags={"team": "search"})
        assert len(records) == 1
        assert records[0].tags["team"] == "search"

    def test_get_total_cost(self, memory_backend):
        """Test getting total cost."""
        for i in range(3):
            memory_backend.save_record(
                CostRecord(
                    timestamp=datetime.now(),
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                    total_cost=1.00,
                )
            )

        total = memory_backend.get_total_cost()
        assert total == 3.00

    def test_get_aggregated_costs(self, memory_backend):
        """Test aggregated costs."""
        # Add records for different models
        memory_backend.save_record(
            CostRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                total_cost=1.00,
            )
        )
        memory_backend.save_record(
            CostRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                total_cost=1.00,
            )
        )
        memory_backend.save_record(
            CostRecord(
                timestamp=datetime.now(),
                provider="anthropic",
                model="claude-3-5-sonnet",
                input_tokens=100,
                output_tokens=50,
                total_cost=2.00,
            )
        )

        # Aggregate by provider
        result = memory_backend.get_aggregated_costs(group_by=["provider"])
        assert "groups" in result
        groups = result["groups"]
        assert len(groups) == 2

    def test_get_report(self, memory_backend, sample_cost_record):
        """Test report generation."""
        memory_backend.save_record(sample_cost_record)

        report = memory_backend.get_report()
        assert report.total_calls == 1
        assert report.total_cost > 0

    def test_delete_records(self, memory_backend):
        """Test deleting records."""
        for i in range(5):
            memory_backend.save_record(
                CostRecord(
                    timestamp=datetime.now(),
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                    total_cost=0.001,
                    tags={"batch": str(i % 2)},
                )
            )

        assert memory_backend.record_count == 5

        # Delete records with specific tag
        deleted = memory_backend.delete_records(tags={"batch": "0"})
        assert deleted == 3
        assert memory_backend.record_count == 2

    def test_max_records_limit(self):
        """Test that old records are evicted when limit is reached."""
        backend = MemoryBackend(max_records=10)

        for i in range(15):
            backend.save_record(
                CostRecord(
                    timestamp=datetime.now(),
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                    total_cost=0.001,
                )
            )

        # Should have evicted some records
        assert backend.record_count <= 10

    def test_health_check(self, memory_backend):
        """Test health check."""
        assert memory_backend.health_check() is True

    def test_clear(self, memory_backend, sample_cost_record):
        """Test clearing all records."""
        memory_backend.save_record(sample_cost_record)
        assert memory_backend.record_count == 1

        memory_backend.clear()
        assert memory_backend.record_count == 0


class TestSQLiteBackend:
    """Tests for SQLiteBackend."""

    @pytest.fixture
    def sqlite_backend(self):
        """Create an in-memory SQLite backend."""
        return SQLiteBackend("sqlite:///:memory:")

    def test_save_record(self, sqlite_backend, sample_cost_record):
        """Test saving a single record."""
        sqlite_backend.save_record(sample_cost_record)

        records = sqlite_backend.get_records()
        assert len(records) == 1

    def test_save_records(self, sqlite_backend):
        """Test saving multiple records."""
        records = [
            CostRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                total_cost=0.001,
            )
            for _ in range(5)
        ]

        sqlite_backend.save_records(records)

        retrieved = sqlite_backend.get_records()
        assert len(retrieved) == 5

    def test_get_total_cost(self, sqlite_backend):
        """Test getting total cost."""
        for _ in range(3):
            sqlite_backend.save_record(
                CostRecord(
                    timestamp=datetime.now(),
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                    total_cost=1.00,
                )
            )

        total = sqlite_backend.get_total_cost()
        assert total == 3.00

    def test_get_report(self, sqlite_backend, sample_cost_record):
        """Test report generation."""
        sqlite_backend.save_record(sample_cost_record)

        report = sqlite_backend.get_report()
        assert report.total_calls == 1
        assert report.total_cost > 0

    def test_health_check(self, sqlite_backend):
        """Test health check."""
        assert sqlite_backend.health_check() is True

    def test_delete_records(self, sqlite_backend):
        """Test deleting records."""
        for i in range(5):
            sqlite_backend.save_record(
                CostRecord(
                    timestamp=datetime.now(),
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                    total_cost=0.001,
                )
            )

        # Delete all records
        deleted = sqlite_backend.delete_records()
        assert deleted == 5

        records = sqlite_backend.get_records()
        assert len(records) == 0
