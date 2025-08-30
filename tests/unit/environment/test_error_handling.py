"""Unit tests for the error handling module."""
import unittest
import time
from unittest.mock import patch, MagicMock
import logging

from adan_trading_bot.environment.error_handling import (
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    TradingError,
    ConfigurationError,
    NetworkError,
    DataError,
    ErrorHandler,
    handle_errors,
)

class TestErrorEnums(unittest.TestCase):
    """Test cases for error enums."""
    
    def test_error_severity(self):
        """Test ErrorSeverity enum values."""
        self.assertEqual(ErrorSeverity.DEBUG.value, 1)
        self.assertEqual(ErrorSeverity.INFO.value, 2)
        self.assertEqual(ErrorSeverity.WARNING.value, 3)
        self.assertEqual(ErrorSeverity.ERROR.value, 4)
        self.assertEqual(ErrorSeverity.CRITICAL.value, 5)
    
    def test_error_category(self):
        """Test ErrorCategory enum values."""
        self.assertEqual(ErrorCategory.CONFIGURATION.value, 1)
        self.assertEqual(ErrorCategory.NETWORK.value, 2)
        self.assertEqual(ErrorCategory.DATA.value, 3)
        self.assertEqual(ErrorCategory.TRADING.value, 4)
        self.assertEqual(ErrorCategory.RESOURCE.value, 5)
        self.assertEqual(ErrorCategory.VALIDATION.value, 6)
        self.assertEqual(ErrorCategory.UNKNOWN.value, 7)

class TestErrorContext(unittest.TestCase):
    """Test cases for ErrorContext class."""
    
    def test_error_context_creation(self):
        """Test ErrorContext initialization."""
        metadata = {"key": "value"}
        context = ErrorContext(
            module="test_module",
            function="test_function",
            metadata=metadata
        )
        
        self.assertEqual(context.module, "test_module")
        self.assertEqual(context.function, "test_function")
        self.assertEqual(context.metadata, metadata)
        self.assertIsInstance(context.timestamp, float)

class TestTradingError(unittest.TestCase):
    """Test cases for TradingError and its subclasses."""
    
    def test_trading_error_creation(self):
        """Test TradingError initialization."""
        context = ErrorContext("test_module", "test_function")
        cause = ValueError("Root cause")
        
        error = TradingError(
            message="Test error",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            cause=cause
        )
        
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.category, ErrorCategory.CONFIGURATION)
        self.assertEqual(error.context, context)
        self.assertEqual(error.cause, cause)
    
    def test_configuration_error(self):
        """Test ConfigurationError initialization."""
        error = ConfigurationError("Config error")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.category, ErrorCategory.CONFIGURATION)
    
    def test_network_error(self):
        """Test NetworkError initialization."""
        error = NetworkError("Network error")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.category, ErrorCategory.NETWORK)
    
    def test_data_error(self):
        """Test DataError initialization."""
        error = DataError("Data error")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.category, ErrorCategory.DATA)

class TestErrorHandler(unittest.TestCase):
    """Test cases for ErrorHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = MagicMock()
        self.handler = ErrorHandler(
            max_retries=2,
            initial_delay=0.1,
            backoff_factor=2.0,
            logger=self.logger
        )
    
    def test_successful_execution(self):
        """Test successful execution without retries."""
        @self.handler.with_retry
        def successful_func():
            return "success"
        
        result = successful_func()
        self.assertEqual(result, "success")
        self.logger.warning.assert_not_called()
    
    def test_retry_and_succeed(self):
        """Test retry logic with eventual success."""
        call_count = 0
        
        @self.handler.with_retry(retry_exceptions=(ValueError,))
        def retry_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = retry_then_succeed()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)
        self.logger.warning.assert_called_once()
    
    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        @self.handler.with_retry(retry_exceptions=(ValueError,))
        def always_fail():
            raise ValueError("Permanent failure")
        
        with self.assertRaises(TradingError) as context:
            always_fail()
        
        self.assertIsInstance(context.exception.cause, ValueError)
        self.assertEqual(self.logger.warning.call_count, 2)  # 2 retries
        self.logger.error.assert_called_once()
    
    @patch('time.sleep')
    def test_retry_delays(self, mock_sleep):
        """Test exponential backoff for retry delays."""
        call_count = 0
        
        @self.handler.with_retry(retry_exceptions=(ValueError,))
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Failure {call_count}")
            return "success"
        
        fail_twice()
        
        # Check that sleep was called with the right delays
        expected_sleep_calls = [
            unittest.mock.call(0.1),  # initial_delay
            unittest.mock.call(0.2),  # initial_delay * backoff_factor
        ]
        mock_sleep.assert_has_calls(expected_sleep_calls)

class TestHandleErrorsDecorator(unittest.TestCase):
    """Test cases for the handle_errors decorator."""
    
    def test_handle_errors_decorator(self):
        """Test the handle_errors decorator factory."""
        call_count = 0
        
        @handle_errors(
            retry_exceptions=(ValueError,),
            max_retries=1,
            initial_delay=0.1,
            backoff_factor=2.0
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = test_func()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)

if __name__ == "__main__":
    unittest.main()
