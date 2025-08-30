"""Unit tests for the circuit breaker module."""
import time
import unittest
from unittest.mock import patch, MagicMock
import logging
from typing import Any, Optional

from adan_trading_bot.environment.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError
)

class TestCircuitBreaker(unittest.TestCase):
    """Test cases for the CircuitBreaker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = MagicMock()
        
        # Create a circuit breaker with low thresholds for testing
        self.cb = CircuitBreaker(
            name="test_service",
            failure_threshold=2,  # Trip after 2 failures
            recovery_timeout=0.1, # Short recovery for testing
            expected_exceptions=(ValueError, ConnectionError),
            logger=self.logger
        )
    
    def test_initial_state(self):
        """Test that the circuit starts in CLOSED state."""
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertEqual(self.cb._failure_count, 0)
        self.assertIsNone(self.cb._last_failure_time)
    
    def test_successful_call(self):
        """Test a successful function call."""
        @self.cb
        def successful_func():
            return "success"
        
        result = successful_func()
        self.assertEqual(result, "success")
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertEqual(self.cb._failure_count, 0)
    
    def test_failure_below_threshold(self):
        """Test failures below the threshold don't open the circuit."""
        @self.cb
        def failing_func():
            raise ConnectionError("Connection failed")
        
        # First failure
        with self.assertRaises(ConnectionError):
            failing_func()
        
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertEqual(self.cb._failure_count, 1)
        
        # Second failure (reaches threshold)
        with self.assertRaises(ConnectionError):
            failing_func()
        
        # Circuit should be open after threshold is reached
        self.assertEqual(self.cb.state, CircuitState.OPEN)
        self.assertEqual(self.cb._failure_count, 2)
    
    def test_circuit_opens_after_threshold(self):
        """Test that the circuit opens after reaching the failure threshold."""
        @self.cb
        def failing_func():
            raise ConnectionError("Connection failed")
        
        # First failure
        with self.assertRaises(ConnectionError):
            failing_func()
        
        # Second failure (reaches threshold)
        with self.assertRaises(ConnectionError):
            failing_func()
        
        # Third attempt should raise CircuitBreakerError
        with self.assertRaises(CircuitBreakerError) as cm:
            failing_func()
        
        self.assertEqual(cm.exception.state, CircuitState.OPEN)
        self.assertIn("is OPEN", str(cm.exception))
        
        # Check logger was called with the right message
        self.logger.error.assert_called()
        self.assertIn("is now OPEN", self.logger.error.call_args[0][0])
    
    def test_circuit_half_open_recovery(self):
        """Test that the circuit can recover after the timeout."""
        # First, make the circuit open
        @self.cb
        def failing_func():
            raise ConnectionError("Connection failed")
        
        for _ in range(2):  # Reach failure threshold
            with self.assertRaises(ConnectionError):
                failing_func()
        
        # Circuit should be open
        self.assertEqual(self.cb.state, CircuitState.OPEN)
        
        # Fast-forward time past the recovery timeout
        with patch('time.monotonic', return_value=time.monotonic() + 1.0):
            # Should now be in HALF_OPEN state
            self.assertEqual(self.cb.state, CircuitState.HALF_OPEN)
            
            # We need multiple successful calls (equal to failure_threshold) to close the circuit
            @self.cb
            def successful_func():
                return "recovered"
            
            # First successful call - circuit should still be HALF_OPEN
            result = successful_func()
            self.assertEqual(result, "recovered")
            self.assertEqual(self.cb.state, CircuitState.HALF_OPEN)
            
            # Second successful call should close the circuit (failure_threshold=2)
            result = successful_func()
            self.assertEqual(result, "recovered")
            self.assertEqual(self.cb.state, CircuitState.CLOSED)
    
    def test_unexpected_exceptions(self):
        """Test that unexpected exceptions don't trigger the circuit breaker."""
        @self.cb
        def unexpected_error():
            raise RuntimeError("Unexpected error")
        
        with self.assertRaises(RuntimeError):
            unexpected_error()
        
        # Circuit should still be closed
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertEqual(self.cb._failure_count, 0)
    
    def test_call_method(self):
        """Test the call() method directly."""
        def test_func():
            return "test"
        
        result = self.cb.call(test_func)
        self.assertEqual(result, "test")
    
    def test_recovery_after_timeout(self):
        """Test that the circuit recovers after the timeout period."""
        # Make the circuit open with expected exceptions
        @self.cb
        def failing_func():
            raise ConnectionError("Connection failed")
            
        for _ in range(2):  # Reach failure threshold
            with self.assertRaises(ConnectionError):
                failing_func()
        
        # Fast-forward time past the recovery timeout
        with patch('time.monotonic', return_value=time.monotonic() + 1.0):
            # Should be in HALF_OPEN state
            self.assertEqual(self.cb.state, CircuitState.HALF_OPEN)
            
            # First successful call - circuit should still be HALF_OPEN
            result = self.cb.call(lambda: "success")
            self.assertEqual(result, "success")
            self.assertEqual(self.cb.state, CircuitState.HALF_OPEN)
            
            # Second successful call should close the circuit (failure_threshold=2)
            result = self.cb.call(lambda: "success")
            self.assertEqual(result, "success")
            self.assertEqual(self.cb.state, CircuitState.CLOSED)
    
    def test_custom_exceptions(self):
        """Test with custom exception types."""
        class CustomError(Exception):
            pass
        
        cb = CircuitBreaker(
            name="custom_error_test",
            failure_threshold=1,
            expected_exceptions=(CustomError,)
        )
        
        @cb
        def raise_custom():
            raise CustomError("Custom error")
        
        # Should be caught and count as a failure
        with self.assertRaises(CustomError):
            raise_custom()
        
        # Next call should trigger circuit open
        with self.assertRaises(CircuitBreakerError):
            raise_custom()
    
    def test_callable_instance(self):
        """Test using CircuitBreaker as a callable instance."""
        def test_func():
            return "test"
        
        result = self.cb(test_func)()
        self.assertEqual(result, "test")

if __name__ == "__main__":
    unittest.main()
