"""
Unit tests for resilience patterns.
"""

import time
import pytest
from unittest.mock import MagicMock, call

from llm_cost_guard.resilience import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    retry_with_backoff,
    RetryConfig,
    ResilientOperation,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_is_closed(self):
        """Test circuit starts closed."""
        breaker = CircuitBreaker()
        
        assert breaker.state == CircuitState.CLOSED

    def test_allows_requests_when_closed(self):
        """Test requests allowed when closed."""
        breaker = CircuitBreaker()
        
        assert breaker.allow_request() is True

    def test_opens_after_threshold_failures(self):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN

    def test_rejects_requests_when_open(self):
        """Test requests rejected when open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

    def test_transitions_to_half_open_after_timeout(self):
        """Test transition to half-open after timeout."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)
        breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN
        
        time.sleep(0.15)
        
        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in half-open."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1, success_threshold=2)
        breaker.record_failure()
        
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN
        
        breaker.record_success()
        breaker.record_success()
        
        assert breaker.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)
        breaker.record_failure()
        
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN
        
        breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()  # Reset
        breaker.record_failure()
        breaker.record_failure()
        
        # Should still be closed (only 2 failures after reset)
        assert breaker.state == CircuitState.CLOSED

    def test_reset(self):
        """Test manual reset."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN
        
        breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED

    def test_excluded_exceptions_not_counted(self):
        """Test excluded exceptions don't count as failures."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            excluded_exceptions=(ValueError,)
        )
        
        breaker.record_failure(ValueError("ignored"))
        
        assert breaker.state == CircuitState.CLOSED

    def test_as_decorator(self):
        """Test circuit breaker as decorator."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        call_count = 0
        
        @breaker
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("failure")
        
        # First failure
        with pytest.raises(Exception):
            failing_function()
        
        # Second failure - opens circuit
        with pytest.raises(Exception):
            failing_function()
        
        # Third call - circuit open
        with pytest.raises(CircuitOpenError):
            failing_function()
        
        # Only 2 actual calls made
        assert call_count == 2

    def test_decorator_records_success(self):
        """Test decorator records success."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        @breaker
        def successful_function():
            return "success"
        
        result = successful_function()
        
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_succeeds_first_try(self):
        """Test no retry on first success."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_function()
        
        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self):
        """Test retry on failure."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("failure")
            return "success"
        
        result = failing_then_success()
        
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        """Test raises after max attempts."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise Exception("always fails")
        
        with pytest.raises(Exception, match="always fails"):
            always_failing()
        
        assert call_count == 3

    def test_only_retries_specified_exceptions(self):
        """Test only retries specified exceptions."""
        call_count = 0
        
        @retry_with_backoff(
            max_attempts=3,
            initial_delay=0.01,
            retryable_exceptions=(ValueError,)
        )
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")
        
        with pytest.raises(TypeError):
            raises_type_error()
        
        # Should not retry for TypeError
        assert call_count == 1

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callbacks = []
        
        def on_retry(exception, attempt):
            callbacks.append((str(exception), attempt))
        
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, initial_delay=0.01, on_retry=on_retry)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"failure {call_count}")
            return "success"
        
        failing_then_success()
        
        assert len(callbacks) == 2
        assert callbacks[0] == ("failure 1", 1)
        assert callbacks[1] == ("failure 2", 2)


class TestResilientOperation:
    """Tests for ResilientOperation."""

    def test_combines_retry_and_circuit_breaker(self):
        """Test combining retry and circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=5)
        retry_config = RetryConfig(max_attempts=2, initial_delay=0.01)
        
        resilient = ResilientOperation(
            circuit_breaker=breaker,
            retry_config=retry_config,
        )
        
        call_count = 0
        
        @resilient
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("transient")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert call_count == 2

    def test_circuit_opens_after_multiple_failures(self):
        """Test circuit opens after multiple call failures."""
        breaker = CircuitBreaker(failure_threshold=2)
        # No retry - each call is one failure
        resilient = ResilientOperation(
            circuit_breaker=breaker,
            retry_config=None,  # Disable retry
        )
        
        @resilient
        def always_fails():
            raise Exception("always fails")
        
        # First failure
        with pytest.raises(Exception):
            always_fails()
        
        # Second failure - opens circuit
        with pytest.raises(Exception):
            always_fails()
        
        # Third call - circuit now open
        with pytest.raises(CircuitOpenError):
            always_fails()

    def test_execute_method(self):
        """Test execute method."""
        breaker = CircuitBreaker(failure_threshold=5)
        resilient = ResilientOperation(circuit_breaker=breaker)
        
        def my_function(x, y):
            return x + y
        
        result = resilient.execute(my_function, 1, 2)
        
        assert result == 3
