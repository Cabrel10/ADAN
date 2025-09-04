#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for data validation and monitoring.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest

# Add the project root to the Python path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from adan_trading_bot.data_processing.observation_validator import (
    ObservationValidator, ValidationLevel, ValidationResult
)


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample market data for testing
        np.random.seed(42)
        self.window_size = 100
        self.n_features = 5
        self.sample_size = self.window_size * 10  # 10 windows of data

        # Create clean data with shape (n_samples, n_features)
        self.clean_data = np.column_stack([
            100 + np.random.randn(self.sample_size).cumsum(),  # open
            105 + np.random.randn(self.sample_size).cumsum(),  # high
            95 + np.random.randn(self.sample_size).cumsum(),   # low
            100 + np.random.randn(self.sample_size).cumsum(),  # close
            1000 + np.random.poisson(100, size=self.sample_size)  # volume
        ])

        # Create data with issues
        self.dirty_data = self.clean_data.copy()

        # Add some missing values (5% of data points)
        mask = np.random.random(self.sample_size * self.n_features) < 0.05
        mask = mask.reshape(self.sample_size, self.n_features)
        self.dirty_data[mask] = np.nan

        # Add some outliers (2% of data points)
        outlier_mask = np.random.random(self.sample_size * self.n_features) < 0.02
        outlier_mask = outlier_mask.reshape(self.sample_size, self.n_features)
        self.dirty_data[outlier_mask] *= 1.5

        # Initialize validator with test config
        self.validator = ObservationValidator({
            'value_range_threshold': 100.0,
            'nan_tolerance': 0.1,  # 10% tolerance for NaNs
            'inf_tolerance': 0.0,  # No tolerance for infinite values
            'timeframes': ['5m'],
            'check_shape': True,
            'check_dtype': True,
            'check_finite': True,
            'check_nan_inf': True,
            'check_value_ranges': True,
            'check_temporal_consistency': True,
            'warn_on_errors': True,
            'raise_on_critical': False
        })

    def test_validate_clean_data(self):
        """Test validation of clean data."""
        # Test with a window of clean data
        window = self.clean_data[:self.window_size]

        # Validate the window
        is_valid, results = self.validator.validate_observation(
            window,
            expected_shape=(self.window_size, self.n_features)
        )

        # Check if there are any validation errors or warnings
        error_messages = [r.message for r in results
                         if r.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
                         and not r.is_valid]

        self.assertTrue(is_valid,
                      f"Validation failed with errors: {error_messages}")

        # Log all validation results for debugging
        for result in results:
            if not result.is_valid:
                print(f"Validation issue: {result.message}")

    def test_validate_dirty_data(self):
        """Test validation of data with quality issues."""
        # Test with a window of dirty data
        window = self.dirty_data[:self.window_size]

        # Validate the window
        is_valid, results = self.validator.validate_observation(
            window,
            expected_shape=(self.window_size, self.n_features)
        )

        # Should detect issues
        self.assertFalse(is_valid)

        # Check specific issues were detected
        issue_messages = [r.message.lower() for r in results if not r.is_valid]
        self.assertTrue(any('nan' in msg.lower() or
                          'out of range' in msg.lower() or
                          'infinite' in msg.lower()
                          for msg in issue_messages))

    def test_validation_with_invalid_shape(self):
        """Test validation with invalid shape."""
        # Create data with wrong shape
        invalid_data = np.random.randn(5, 3)  # Wrong shape

        # Validate with strict shape checking
        is_valid, results = self.validator.validate_observation(
            invalid_data,
            expected_shape=(self.window_size, self.n_features)
        )

        # Should fail due to shape mismatch
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid shape" in r.message for r in results))

    def test_validation_with_constant_data(self):
        """Test validation with constant data that should trigger warnings."""
        # Create constant data (zero variance)
        constant_data = np.ones((self.window_size, self.n_features))

        # Validate with statistics checking enabled
        is_valid, results = self.validator.validate_observation(
            constant_data,
            expected_shape=(self.window_size, self.n_features)
        )

        # Should have warnings about low variance
        self.assertTrue(
            any("low variance" in r.message.lower() for r in results)
        )
        self.assertTrue(
            any("zero variance" in r.message.lower() for r in results)
        )

    def test_validation_with_duplicate_sequences(self):
        """Test validation with duplicate sequences that should trigger warnings."""
        # Create data with duplicate sequences
        base_data = np.random.randn(5, self.n_features)
        data_with_duplicates = np.vstack([base_data] * 10)  # Repeat the same data

        # Validate with temporal consistency checking
        is_valid, results = self.validator.validate_observation(
            data_with_duplicates,
            expected_shape=(len(data_with_duplicates), self.n_features)
        )

        # Should have warnings about consecutive identical time steps
        self.assertTrue(
            any("consecutive identical" in r.message.lower()
                for r in results)
        )

    def test_batch_validation(self):
        """Test validation of a batch of observations."""
        # Create a batch of observations
        batch_size = 3
        batch_data = np.random.randn(
            batch_size,
            self.window_size,
            self.n_features
        )

        # Validate batch
        all_valid, batch_results = self.validator.validate_batch(batch_data)

        # Check results
        self.assertTrue(isinstance(all_valid, bool))
        self.assertEqual(len(batch_results), batch_size)
        for results in batch_results:
            self.assertTrue(
                all(isinstance(r, ValidationResult) for r in results)
            )

    def test_custom_configuration(self):
        """Test validator with custom configuration."""
        # Create validator with custom config
        config = {
            'check_shape': True,
            'check_dtype': False,  # Disable dtype checking
            'value_range_threshold': 50.0,
            'timeframes': ['15m', '1h', '4h']
        }
        validator = ObservationValidator(config=config)

        # Test with data that would fail dtype check if enabled
        test_data = np.random.randint(
            0, 10, (self.window_size, self.n_features)
        )
        is_valid, _ = validator.validate_observation(
            test_data,
            expected_shape=(self.window_size, self.n_features)
        )

        # Should pass because dtype checking is disabled
        self.assertTrue(is_valid)

    def test_validation_report_export(self):
        """Test exporting validation results to a report file."""
        # Create test data with some issues
        test_data = np.random.randn(self.window_size, self.n_features)
        test_data[0, 0] = np.nan  # Add a NaN value

        # Run validation
        _, results = self.validator.validate_observation(
            test_data,
            expected_shape=(self.window_size, self.n_features)
        )

        # Export to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            report_path = tmp_file.name

        try:
            # Save report
            self.validator.save_validation_report(results, report_path)

            # Verify file was created
            self.assertTrue(os.path.exists(report_path))
            self.assertGreater(os.path.getsize(report_path), 0)

        finally:
            # Clean up
            if os.path.exists(report_path):
                os.unlink(report_path)

    def test_reset_statistics(self):
        """Test resetting validation statistics."""
        # Run some validations
        for _ in range(3):
            self.validator.validate_observation(
                np.random.randn(self.window_size, self.n_features)
            )

        # Reset stats
        self.validator.reset_stats()

        # Check that stats were reset
        stats = self.validator.get_validation_summary()
        self.assertEqual(stats['total_validations'], 0)
        self.assertEqual(stats['stats']['passed_validations'], 0)
        self.assertEqual(stats['stats']['failed_validations'], 0)

    def test_logging_validation_results(self):
        """Test logging of validation results."""
        # Create test data with some issues
        test_data = np.random.randn(self.window_size, self.n_features)
        test_data[0, 0] = np.nan  # Add a NaN value

        # Run validation with logging
        with self.assertLogs(level='WARNING') as cm:
            _, results = self.validator.validate_observation(
                test_data,
                expected_shape=(self.window_size, self.n_features)
            )
            self.validator.log_validation_results(
                results, "test_observation"
            )

        # Check that logs were generated
        self.assertTrue(any("NaN values" in log for log in cm.output))

    def test_validation_statistics(self):
        """Test validation statistics tracking."""
        # Reset statistics
        self.validator.reset_stats()

        # Run multiple validations
        for i in range(5):
            window = self.clean_data[i*self.window_size:(i+1)*self.window_size]
            is_valid, _ = self.validator.validate_observation(
                window,
                expected_shape=(self.window_size, self.n_features)
            )
            # Ensure we're tracking the validation result
            self.assertTrue(is_valid, f"Validation {i} failed")

        # Check statistics
        stats = self.validator.get_validation_summary()
        self.assertEqual(stats['total_validations'], 5)
        self.assertEqual(stats['stats']['passed_validations'], 5)
        self.assertEqual(stats['stats']['failed_validations'], 0)

    def test_validation_with_different_dtypes(self):
        """Test validation with different data types."""
        # Test with float32 (should pass)
        test_data = self.clean_data[:self.window_size].astype(np.float32)
        is_valid, results = self.validator.validate_observation(
            test_data,
            expected_shape=(self.window_size, self.n_features),
            expected_dtype=np.float32
        )
        self.assertTrue(is_valid,
                      f"Validation failed for float32: "
                      f"{[r.message for r in results if not r.is_valid]}")

        # Test with float64 (should pass as it's a floating type)
        test_data = self.clean_data[:self.window_size].astype(np.float64)
        is_valid, results = self.validator.validate_observation(
            test_data,
            expected_shape=(self.window_size, self.n_features),
            expected_dtype=np.float32
        )
        self.assertTrue(is_valid,
                      f"Validation failed for float64: "
                      f"{[r.message for r in results if not r.is_valid]}")


if __name__ == '__main__':
    unittest.main()
