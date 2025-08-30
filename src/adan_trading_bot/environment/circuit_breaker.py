"""
Circuit Breaker pattern implementation for external service calls.

This module provides a circuit breaker to prevent cascading failures when external
services are unavailable or experiencing issues.
"""
import time
import logging
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar, Generic, Type
from functools import wraps

T = TypeVar('T')

class CircuitState(Enum):
    """Possible states of the circuit breaker."""
    CLOSED = auto()    # Normal operation, all requests pass through
    OPEN = auto()      # Circuit is open, all requests fail immediately
    HALF_OPEN = auto() # Test state to check if service has recovered

class CircuitBreakerError(Exception):
    """Exception raised when the circuit is open."""
    def __init__(self, name: str, state: CircuitState, retry_after: Optional[float] = None):
        self.name = name
        self.state = state
        self.retry_after = retry_after
        message = f"Circuit '{name}' is {state.name}"
        if retry_after:
            message += f". Retry after {retry_after:.1f} seconds"
        super().__init__(message)

class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation to handle external service failures.
    
    Implements the circuit breaker pattern to detect failures and encapsulate the logic
    of preventing a service from constantly trying to execute an operation that's likely
    to fail.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: tuple[Type[Exception], ...] = (Exception,),
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker for identification
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before attempting to close the circuit
            expected_exceptions: Exceptions that should be treated as failures
            logger: Logger instance (creates one if None)
        """
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(0.1, recovery_timeout)
        self.expected_exceptions = expected_exceptions
        self.logger = logger or logging.getLogger(__name__)
        
        # Circuit state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._success_count = 0
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit."""
        # Check if we should transition from OPEN to HALF_OPEN
        if (self._state == CircuitState.OPEN and 
                self._last_failure_time is not None and
                time.monotonic() > self._last_failure_time + self.recovery_timeout):
            self._state = CircuitState.HALF_OPEN
            self.logger.warning(
                f"Circuit '{self.name}' is now HALF_OPEN. Testing service recovery..."
            )
        return self._state
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap a function with circuit breaker logic."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute the function with circuit breaker logic.
        
        Args:
            func: The function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
            
        Raises:
            CircuitBreakerError: If the circuit is open
            Exception: Any exception raised by the wrapped function
        """
        # Check circuit state
        current_state = self.state
        
        if current_state == CircuitState.OPEN:
            retry_after = (self._last_failure_time or 0) + self.recovery_timeout - time.monotonic()
            raise CircuitBreakerError(
                self.name, 
                CircuitState.OPEN, 
                max(0, retry_after)
            )
        
        # Try to execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exceptions as e:
            self._on_failure()
            raise  # Re-raise the original exception
            
        except Exception as e:
            # Log unexpected exceptions but don't count them as failures
            self.logger.error(
                f"Unexpected error in circuit '{self.name}': {str(e)}",
                exc_info=True
            )
            raise
    
    def _on_success(self):
        """Handle a successful function call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            
            # If we've had enough successful calls, close the circuit
            if self._success_count >= self.failure_threshold:
                self._close()
        
        # Reset failure count on success
        self._failure_count = 0
    
    def _on_failure(self):
        """Handle a failed function call."""
        self._failure_count += 1
        self._success_count = 0
        self._last_failure_time = time.monotonic()
        
        # Check if we should open the circuit
        if (self._state == CircuitState.CLOSED and 
                self._failure_count >= self.failure_threshold):
            self._open()
        elif self._state == CircuitState.HALF_OPEN:
            # If we fail in half-open state, go back to open
            self._open()
    
    def _open(self):
        """Open the circuit."""
        old_state = self._state
        self._state = CircuitState.OPEN
        
        if old_state != CircuitState.OPEN:
            self.logger.error(
                f"Circuit '{self.name}' is now OPEN. "
                f"Failed {self._failure_count} times. "
                f"Will retry after {self.recovery_timeout} seconds."
            )
    
    def _close(self):
        """Close the circuit."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        
        if old_state != CircuitState.CLOSED:
            self.logger.info(f"Circuit '{self.name}' is now CLOSED. Service is healthy.")

# Example usage:
if __name__ == "__main__":
    import random
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a circuit breaker
    cb = CircuitBreaker["example"](
        name="example_service",
        failure_threshold=3,
        recovery_timeout=5.0,
        expected_exceptions=(ConnectionError, TimeoutError)
    )
    
    # Function that might fail
    @cb
    def unreliable_service():
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Service unavailable")
        return "Success!"
    
    # Test the circuit breaker
    for i in range(10):
        try:
            print(f"Attempt {i+1}:", end=" ")
            result = unreliable_service()
            print(result)
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1)
