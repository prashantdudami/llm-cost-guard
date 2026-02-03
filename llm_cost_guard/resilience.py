"""
Resilience patterns for LLM Cost Guard.

Implements circuit breaker, retry with backoff, and other resilience patterns.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Type, Tuple

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 30.0  # Seconds to wait before half-open
    excluded_exceptions: Tuple[Type[Exception], ...] = ()  # Don't count these as failures


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascade failures by stopping requests to a failing service.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered
    
    Usage:
        breaker = CircuitBreaker(failure_threshold=5, timeout=30)
        
        @breaker
        def call_redis():
            return redis.get("key")
        
        # Or manually:
        if breaker.allow_request():
            try:
                result = call_redis()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        excluded_exceptions: Tuple[Type[Exception], ...] = (),
        name: str = "default",
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes to close from half-open
            timeout: Seconds to wait in open state before testing
            excluded_exceptions: Exception types that don't count as failures
            name: Name for logging
        """
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout = timeout
        self._excluded_exceptions = excluded_exceptions
        self._name = name
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self._timeout:
                logger.info(f"Circuit breaker '{self._name}' entering half-open state")
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                return True  # Allow test request
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    logger.info(f"Circuit breaker '{self._name}' closing (recovered)")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed request."""
        # Check if exception is excluded
        if exception and isinstance(exception, self._excluded_exceptions):
            return
        
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker '{self._name}' reopening (test failed)")
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    logger.warning(f"Circuit breaker '{self._name}' opening (threshold reached)")
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.allow_request():
                raise CircuitOpenError(
                    f"Circuit breaker '{self._name}' is open",
                    breaker=self,
                )
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        return wrapper


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, breaker: CircuitBreaker):
        super().__init__(message)
        self.breaker = breaker


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator for retry with exponential backoff.
    
    Cloud-agnostic implementation using only standard library.
    
    Usage:
        @retry_with_backoff(max_attempts=3, initial_delay=1.0)
        def call_api():
            return requests.get("https://api.example.com")
    
    Args:
        max_attempts: Maximum number of attempts (must be >= 1)
        initial_delay: Initial delay in seconds (must be > 0)
        max_delay: Maximum delay in seconds (must be > 0)
        exponential_base: Base for exponential backoff (must be > 1)
        jitter: Add random jitter to delays
        retryable_exceptions: Exception types to retry
        on_retry: Callback called on each retry (exception, attempt)
    """
    import secrets as secrets_module
    
    # Validate parameters
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if initial_delay <= 0:
        raise ValueError("initial_delay must be > 0")
    if max_delay <= 0:
        raise ValueError("max_delay must be > 0")
    if exponential_base <= 1:
        raise ValueError("exponential_base must be > 1")
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )
                    
                    # Add jitter using cryptographically secure random
                    if jitter:
                        # Generate random float between 0.5 and 1.0
                        jitter_factor = 0.5 + (secrets_module.randbelow(1000) / 2000.0)
                        delay = delay * jitter_factor
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(delay)
            
            raise last_exception  # Should never reach here
        
        return wrapper
    
    return decorator


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class ResilientOperation:
    """
    Combines circuit breaker and retry for resilient operations.
    
    Usage:
        resilient = ResilientOperation(
            circuit_breaker=CircuitBreaker(failure_threshold=5),
            retry_config=RetryConfig(max_attempts=3),
        )
        
        @resilient
        def call_redis():
            return redis.get("key")
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize resilient operation.
        
        Args:
            circuit_breaker: Circuit breaker instance
            retry_config: Retry configuration
        """
        self._breaker = circuit_breaker
        self._retry_config = retry_config or RetryConfig()

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        # Apply retry first (inner decorator)
        if self._retry_config:
            func = retry_with_backoff(
                max_attempts=self._retry_config.max_attempts,
                initial_delay=self._retry_config.initial_delay,
                max_delay=self._retry_config.max_delay,
                exponential_base=self._retry_config.exponential_base,
                jitter=self._retry_config.jitter,
            )(func)
        
        # Apply circuit breaker (outer decorator)
        if self._breaker:
            func = self._breaker(func)
        
        return func

    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function with resilience patterns."""
        wrapped = self(func)
        return wrapped(*args, **kwargs)
