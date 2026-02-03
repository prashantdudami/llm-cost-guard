"""
Unit tests for Redis backend.

These tests mock the Redis client to avoid requiring a running Redis instance.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

from llm_cost_guard.models import CostRecord, ModelType

# Skip all tests if redis is not installed
try:
    import redis
    REDIS_INSTALLED = True
except ImportError:
    REDIS_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not REDIS_INSTALLED, 
    reason="redis package not installed"
)


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.setex.return_value = True
    mock_client.incrbyfloat.return_value = "10.5"
    mock_client.hincrby.return_value = 1
    mock_client.hincrbyfloat.return_value = "10.5"
    mock_client.expire.return_value = True
    mock_client.delete.return_value = 1
    mock_client.zrangebyscore.return_value = []
    mock_client.zadd.return_value = 1
    mock_client.pipeline.return_value = MagicMock()
    mock_client.pipeline.return_value.execute.return_value = []
    
    # Mock script registration
    mock_script = MagicMock()
    mock_client.register_script.return_value = mock_script
    
    return mock_client


@pytest.fixture
def redis_backend(mock_redis_client):
    """Create a Redis backend with mocked client."""
    with patch("redis.from_url", return_value=mock_redis_client):
        from llm_cost_guard.backends.redis_backend import RedisBackend
        backend = RedisBackend(url="redis://localhost:6379/0")
        return backend


class TestRedisBackendInit:
    """Tests for Redis backend initialization."""

    def test_init_with_url(self, mock_redis_client):
        """Test initialization with URL."""
        with patch("redis.from_url", return_value=mock_redis_client) as mock_from_url:
            from llm_cost_guard.backends.redis_backend import RedisBackend
            backend = RedisBackend(url="redis://localhost:6379/0")
            
            mock_from_url.assert_called_once()
            assert backend._prefix == "llm_cost_guard:"

    def test_init_with_custom_prefix(self, mock_redis_client):
        """Test initialization with custom prefix."""
        with patch("redis.from_url", return_value=mock_redis_client):
            from llm_cost_guard.backends.redis_backend import RedisBackend
            backend = RedisBackend(
                url="redis://localhost:6379/0",
                prefix="my_app:",
            )
            
            assert backend._prefix == "my_app:"

    def test_init_registers_lua_scripts(self, mock_redis_client):
        """Test that Lua scripts are registered."""
        with patch("redis.from_url", return_value=mock_redis_client):
            from llm_cost_guard.backends.redis_backend import RedisBackend
            backend = RedisBackend(url="redis://localhost:6379/0")
            
            # Should register 3 scripts (check, reserve, finalize)
            assert mock_redis_client.register_script.call_count == 3


class TestRedisBackendBudget:
    """Tests for Redis budget operations."""

    def test_check_budget_atomic_allowed(self, redis_backend, mock_redis_client):
        """Test atomic budget check when allowed."""
        # Mock script to return success
        redis_backend._budget_check_script.return_value = [50.0, 40.0, 0]  # [new_total, current, warning]
        
        allowed, current, is_warning = redis_backend.check_budget_atomic(
            budget_name="daily",
            amount=10.0,
            limit=100.0,
            period_seconds=86400,
        )
        
        assert allowed is True
        assert current == 50.0
        assert is_warning is False

    def test_check_budget_atomic_exceeded(self, redis_backend):
        """Test atomic budget check when exceeded."""
        # Mock script to return exceeded
        redis_backend._budget_check_script.return_value = [-1, 95.0, 100.0]
        
        allowed, current, is_warning = redis_backend.check_budget_atomic(
            budget_name="daily",
            amount=10.0,
            limit=100.0,
            period_seconds=86400,
        )
        
        assert allowed is False
        assert current == 95.0

    def test_check_budget_atomic_warning(self, redis_backend):
        """Test atomic budget check with warning."""
        # Mock script to return warning
        redis_backend._budget_check_script.return_value = [85.0, 75.0, 1]
        
        allowed, current, is_warning = redis_backend.check_budget_atomic(
            budget_name="daily",
            amount=10.0,
            limit=100.0,
            period_seconds=86400,
            warning_threshold=0.8,
        )
        
        assert allowed is True
        assert current == 85.0
        assert is_warning is True

    def test_reserve_budget_success(self, redis_backend):
        """Test budget reservation success."""
        redis_backend._budget_reserve_script.return_value = [60.0, 50.0, 0]
        
        allowed, effective = redis_backend.reserve_budget(
            budget_name="daily",
            estimated_amount=10.0,
            limit=100.0,
            reservation_id="res-123",
            period_seconds=86400,
        )
        
        assert allowed is True
        assert effective == 60.0

    def test_reserve_budget_exceeded(self, redis_backend):
        """Test budget reservation when would exceed."""
        redis_backend._budget_reserve_script.return_value = [-1, 95.0, 100.0]
        
        allowed, effective = redis_backend.reserve_budget(
            budget_name="daily",
            estimated_amount=10.0,
            limit=100.0,
            reservation_id="res-123",
            period_seconds=86400,
        )
        
        assert allowed is False
        assert effective == 95.0

    def test_finalize_budget(self, redis_backend):
        """Test budget finalization."""
        redis_backend._budget_finalize_script.return_value = "55.0"
        
        new_total = redis_backend.finalize_budget(
            budget_name="daily",
            reserved_amount=10.0,
            actual_amount=8.0,  # Actual was less than reserved
            reservation_id="res-123",
            period_seconds=86400,
        )
        
        assert new_total == 55.0

    def test_release_reservation(self, redis_backend, mock_redis_client):
        """Test releasing a reservation."""
        pipe = MagicMock()
        mock_redis_client.pipeline.return_value = pipe
        
        redis_backend.release_reservation(
            budget_name="daily",
            reserved_amount=10.0,
            reservation_id="res-123",
        )
        
        pipe.incrbyfloat.assert_called()
        pipe.execute.assert_called()

    def test_get_budget_spending(self, redis_backend, mock_redis_client):
        """Test getting budget spending."""
        mock_redis_client.get.return_value = "75.5"
        
        spending = redis_backend.get_budget_spending("daily")
        
        assert spending == 75.5

    def test_get_budget_spending_no_budget(self, redis_backend, mock_redis_client):
        """Test getting spending for non-existent budget."""
        mock_redis_client.get.return_value = None
        
        spending = redis_backend.get_budget_spending("nonexistent")
        
        assert spending == 0.0

    def test_reset_budget(self, redis_backend, mock_redis_client):
        """Test resetting a budget."""
        redis_backend.reset_budget("daily")
        
        mock_redis_client.delete.assert_called()


class TestRedisBackendRecords:
    """Tests for Redis record storage."""

    def test_save_record(self, redis_backend, mock_redis_client):
        """Test saving a cost record."""
        pipe = MagicMock()
        mock_redis_client.pipeline.return_value = pipe
        
        record = CostRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            total_cost=0.01,
            tags={"team": "search"},
        )
        
        redis_backend.save_record(record)
        
        pipe.setex.assert_called()
        pipe.zadd.assert_called()
        pipe.execute.assert_called()

    def test_get_records_empty(self, redis_backend, mock_redis_client):
        """Test getting records when none exist."""
        mock_redis_client.zrangebyscore.return_value = []
        
        records = redis_backend.get_records()
        
        assert records == []

    def test_get_records_with_date_filter(self, redis_backend, mock_redis_client):
        """Test getting records with date filter."""
        mock_redis_client.zrangebyscore.return_value = []
        
        start = datetime.now() - timedelta(days=1)
        end = datetime.now()
        
        records = redis_backend.get_records(start_date=start, end_date=end)
        
        # Verify zrangebyscore was called with correct score range
        mock_redis_client.zrangebyscore.assert_called_once()


class TestRedisBackendHealth:
    """Tests for Redis health operations."""

    def test_health_check_healthy(self, redis_backend, mock_redis_client):
        """Test health check when healthy."""
        mock_redis_client.ping.return_value = True
        
        assert redis_backend.health_check() is True

    def test_health_check_unhealthy(self, redis_backend, mock_redis_client):
        """Test health check when unhealthy."""
        mock_redis_client.ping.side_effect = Exception("Connection refused")
        
        assert redis_backend.health_check() is False

    def test_get_metrics(self, redis_backend, mock_redis_client):
        """Test getting metrics."""
        mock_redis_client.ping.return_value = True
        
        metrics = redis_backend.get_metrics()
        
        assert "backend_failures" in metrics
        assert "connected" in metrics
        assert metrics["connected"] is True

    def test_close(self, redis_backend, mock_redis_client):
        """Test closing connection."""
        redis_backend.close()
        
        mock_redis_client.close.assert_called_once()


class TestRedisBackendErrorHandling:
    """Tests for Redis error handling."""

    def test_check_budget_redis_error(self, redis_backend):
        """Test handling of Redis errors during budget check."""
        redis_backend._budget_check_script.side_effect = Exception("Redis error")
        
        with pytest.raises(Exception):
            redis_backend.check_budget_atomic(
                budget_name="daily",
                amount=10.0,
                limit=100.0,
                period_seconds=86400,
            )
        
        # Should increment failure metric
        assert redis_backend._metrics["backend_failures"] == 1

    def test_reserve_budget_redis_error(self, redis_backend):
        """Test handling of Redis errors during reservation."""
        redis_backend._budget_reserve_script.side_effect = Exception("Redis error")
        
        with pytest.raises(Exception):
            redis_backend.reserve_budget(
                budget_name="daily",
                estimated_amount=10.0,
                limit=100.0,
                reservation_id="res-123",
                period_seconds=86400,
            )
        
        assert redis_backend._metrics["backend_failures"] == 1

    def test_save_record_redis_error(self, redis_backend, mock_redis_client):
        """Test handling of Redis errors during record save."""
        pipe = MagicMock()
        pipe.execute.side_effect = Exception("Redis error")
        mock_redis_client.pipeline.return_value = pipe
        
        record = CostRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            total_cost=0.01,
        )
        
        with pytest.raises(Exception):
            redis_backend.save_record(record)
        
        assert redis_backend._metrics["backend_failures"] == 1
