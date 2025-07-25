#!/usr/bin/env python3
"""
Test script for observation validation functionality.

This script tests the comprehensive observation validation system to ensure
observations meet quality standards and design specifications.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
import traceback

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.data_processing.observation_validator import ObservationValidator, ValidationLevel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_observations():
    """Create various test observations for validation testing."""
    
    # Valid observation
    valid_obs = np.random.normal(0, 1, (3, 100, 6))
    
    # Invalid shape observation
    invalid_shape_obs = np.random.normal(0, 1, (2, 50, 4))
    
    # Observation with NaN values
    nan_obs = np.random.normal(0, 1, (3, 100, 6))
    nan_obs[0, 10:15, 2] = np.nan
    
    # Observation with infinite values
    inf_obs = np.random.normal(0, 1, (3, 100, 6))
    inf_obs[1, 20:25, 1] = np.inf
    
    # Observation with extreme values
    extreme_obs = np.random.normal(0, 1, (3, 100, 6))
    extreme_obs[2, 30:35, 3] = 1000.0
    
    # Observation with constant values (zero variance)
    constant_obs = np.random.normal(0, 1, (3, 100, 6))
    constant_obs[0, :, 0] = 5.0  # Make first feature constant
    
    # Observation with repeated patterns
    repeated_obs = np.random.normal(0, 1, (3, 100, 6))
    # Create obvious repeated patterns - make many consecutive rows identical
    pattern = repeated_obs[1, 0, :]  # Single row pattern
    for i in range(20, 40):  # 20 consecutive identical rows
        repeated_obs[1, i, :] = pattern
    
    return {
        'valid': valid_obs,
        'invalid_shape': invalid_shape_obs,
        'with_nan': nan_obs,
        'with_inf': inf_obs,
        'extreme_values': extreme_obs,
        'constant_values': constant_obs,
        'repeated_patterns': repeated_obs
    }

def test_basic_validation():
    """Test basic observation validation functionality."""
    logger.info("🔍 Testing basic observation validation...")
    
    # Initialize validator
    validator = ObservationValidator(
        expected_shape=(3, 100, 6),
        timeframes=['5m', '1h', '4h'],
        value_range_threshold=10.0,
        nan_tolerance=0.0,
        inf_tolerance=0.0
    )
    
    # Create test observations
    test_obs = create_test_observations()
    
    # Test 1: Valid observation
    logger.info("📊 Test 1: Validating correct observation...")
    is_valid, results = validator.validate_observation(test_obs['valid'])
    
    logger.info(f"   Valid observation result: {is_valid}")
    info_count = sum(1 for r in results if r.level == ValidationLevel.INFO)
    warning_count = sum(1 for r in results if r.level == ValidationLevel.WARNING)
    error_count = sum(1 for r in results if r.level == ValidationLevel.ERROR)
    
    logger.info(f"   Results: {info_count} info, {warning_count} warnings, {error_count} errors")
    
    assert is_valid, "Valid observation should pass validation"
    logger.info("   ✅ Valid observation test passed")
    
    # Test 2: Invalid shape observation
    logger.info("📏 Test 2: Validating observation with wrong shape...")
    is_valid, results = validator.validate_observation(test_obs['invalid_shape'])
    
    logger.info(f"   Invalid shape observation result: {is_valid}")
    error_count = sum(1 for r in results if r.level == ValidationLevel.ERROR)
    
    assert not is_valid, "Invalid shape observation should fail validation"
    assert error_count > 0, "Should have shape errors"
    logger.info("   ✅ Invalid shape test passed")
    
    # Test 3: NaN values observation
    logger.info("🚫 Test 3: Validating observation with NaN values...")
    is_valid, results = validator.validate_observation(test_obs['with_nan'])
    
    logger.info(f"   NaN observation result: {is_valid}")
    error_count = sum(1 for r in results if r.level == ValidationLevel.ERROR)
    
    assert not is_valid, "Observation with NaN should fail validation"
    assert error_count > 0, "Should have NaN errors"
    logger.info("   ✅ NaN values test passed")
    
    # Test 4: Infinite values observation
    logger.info("♾️ Test 4: Validating observation with infinite values...")
    is_valid, results = validator.validate_observation(test_obs['with_inf'])
    
    logger.info(f"   Infinite values observation result: {is_valid}")
    error_count = sum(1 for r in results if r.level == ValidationLevel.ERROR)
    
    assert not is_valid, "Observation with infinite values should fail validation"
    assert error_count > 0, "Should have infinite value errors"
    logger.info("   ✅ Infinite values test passed")
    
    # Test 5: Extreme values observation
    logger.info("⚡ Test 5: Validating observation with extreme values...")
    is_valid, results = validator.validate_observation(test_obs['extreme_values'])
    
    logger.info(f"   Extreme values observation result: {is_valid}")
    warning_count = sum(1 for r in results if r.level == ValidationLevel.WARNING)
    
    # Should pass validation but generate warnings
    assert warning_count > 0, "Should have warnings for extreme values"
    logger.info("   ✅ Extreme values test passed")
    
    return True

def test_statistical_validation():
    """Test statistical validation features."""
    logger.info("📈 Testing statistical validation...")
    
    validator = ObservationValidator(
        expected_shape=(3, 100, 6),
        timeframes=['5m', '1h', '4h']
    )
    
    test_obs = create_test_observations()
    
    # Test constant values detection
    logger.info("📊 Test: Constant values detection...")
    is_valid, results = validator.validate_observation(test_obs['constant_values'], strict=False)
    
    warning_count = sum(1 for r in results if r.level == ValidationLevel.WARNING and 'variance' in r.message.lower())
    logger.info(f"   Constant values warnings: {warning_count}")
    
    # Should detect low variance
    assert warning_count > 0, "Should detect constant values"
    logger.info("   ✅ Constant values detection passed")
    
    # Test repeated patterns detection
    logger.info("🔄 Test: Repeated patterns detection...")
    is_valid, results = validator.validate_observation(test_obs['repeated_patterns'], strict=False)
    
    temporal_warnings = sum(1 for r in results if r.level == ValidationLevel.WARNING and 'consecutive' in r.message.lower())
    logger.info(f"   Temporal consistency warnings: {temporal_warnings}")
    
    # Should detect repeated patterns
    assert temporal_warnings > 0, "Should detect repeated patterns"
    logger.info("   ✅ Repeated patterns detection passed")
    
    return True

def test_batch_validation():
    """Test batch validation functionality."""
    logger.info("📦 Testing batch validation...")
    
    validator = ObservationValidator(expected_shape=(3, 100, 6))
    
    # Create a batch of observations
    batch_size = 5
    batch_obs = np.random.normal(0, 1, (batch_size, 3, 100, 6))
    
    # Add some issues to specific observations
    batch_obs[1, 0, 10:15, 2] = np.nan  # Add NaN to second observation
    batch_obs[3, 2, 20:25, 1] = np.inf  # Add inf to fourth observation
    
    # Validate the batch
    all_valid, batch_results = validator.validate_batch(batch_obs, strict=True)
    
    logger.info(f"   Batch validation result: {all_valid}")
    logger.info(f"   Individual results count: {len(batch_results)}")
    
    # Should fail because of NaN and inf values
    assert not all_valid, "Batch with issues should fail validation"
    assert len(batch_results) == batch_size, "Should have results for each observation"
    
    # Check that problematic observations are detected
    obs_1_errors = sum(1 for r in batch_results[1] if r.level == ValidationLevel.ERROR)
    obs_3_errors = sum(1 for r in batch_results[3] if r.level == ValidationLevel.ERROR)
    
    assert obs_1_errors > 0, "Observation 1 should have errors (NaN)"
    assert obs_3_errors > 0, "Observation 3 should have errors (inf)"
    
    logger.info("   ✅ Batch validation test passed")
    
    return True

def test_with_state_builder():
    """Test validation integration with StateBuilder."""
    logger.info("🔧 Testing validation with StateBuilder...")
    
    # Create synthetic data
    timeframes = ['5m', '1h', '4h']
    synthetic_data = {}
    
    for tf in timeframes:
        n_samples = 200
        data = {
            'open': np.random.uniform(100, 200, n_samples),
            'high': np.random.uniform(150, 250, n_samples),
            'low': np.random.uniform(50, 150, n_samples),
            'close': np.random.uniform(100, 200, n_samples),
            'volume': np.random.uniform(1000, 10000, n_samples),
            'minutes_since_update': np.random.randint(0, 60, n_samples)
        }
        
        # Ensure price consistency
        for i in range(n_samples):
            data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
            data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
        
        synthetic_data[tf] = pd.DataFrame(data)
    
    # Initialize StateBuilder and validator
    state_builder = StateBuilder(
        window_size=100,
        normalize=True
    )
    
    validator = ObservationValidator(
        expected_shape=(3, 100, 6),
        timeframes=timeframes
    )
    
    # Fit scalers
    state_builder.fit_scalers(synthetic_data)

    # --- Dimension Validation ---
    logger.info("📏 Validating state dimension...")
    try:
        state_builder.validate_dimension(synthetic_data)
        logger.info("   ✅ State dimension validation successful.")
    except ValueError as e:
        logger.error(f"   ❌ State dimension validation failed: {e}")
        raise e
    # --- End Validation ---
    
    # Generate observations and validate them
    current_idx = 150
    
    # Test individual observations
    observations = state_builder.build_observation(current_idx, synthetic_data)
    
    for tf, obs in observations.items():
        logger.info(f"   Validating {tf} observation shape: {obs.shape}")
        
        # Create a mock 3D observation for validation (validator expects 3D)
        mock_3d_obs = np.expand_dims(obs, axis=0)
        mock_3d_obs = np.repeat(mock_3d_obs, 3, axis=0)
        
        is_valid, results = validator.validate_observation(mock_3d_obs, strict=False)
        logger.info(f"   {tf} validation result: {is_valid}")
    
    # Test multi-channel observation
    multi_obs = state_builder.build_multi_channel_observation(current_idx, synthetic_data)
    
    if multi_obs is not None:
        logger.info(f"   Multi-channel observation shape: {multi_obs.shape}")
        
        is_valid, results = validator.validate_observation(multi_obs, strict=False)
        logger.info(f"   Multi-channel validation result: {is_valid}")
        
        # Log validation results
        validator.log_validation_results(results, "multi_channel_test")
        
        # Should be valid
        assert is_valid or len([r for r in results if r.level == ValidationLevel.ERROR]) == 0, "StateBuilder observation should be valid"
        
        logger.info("   ✅ StateBuilder integration test passed")
    else:
        logger.error("   ❌ Multi-channel observation is None")
        return False
    
    return True

def test_validation_statistics():
    """Test validation statistics tracking."""
    logger.info("📊 Testing validation statistics...")
    
    validator = ObservationValidator(expected_shape=(3, 100, 6))
    
    # Perform multiple validations
    test_obs = create_test_observations()
    
    # Validate different types of observations
    validator.validate_observation(test_obs['valid'])
    validator.validate_observation(test_obs['with_nan'])
    validator.validate_observation(test_obs['with_inf'])
    validator.validate_observation(test_obs['extreme_values'], strict=False)
    
    # Get statistics
    stats = validator.get_validation_summary()
    
    logger.info(f"   Validation statistics: {stats}")
    
    assert stats['total_validations'] == 4, "Should have 4 total validations"
    assert stats['success_rate'] > 0, "Should have some successful validations"
    assert stats['failure_rate'] > 0, "Should have some failed validations"
    
    logger.info("   ✅ Validation statistics test passed")
    
    return True

def main():
    """Main test function."""
    logger.info("🚀 Starting observation validation tests...")
    
    try:
        # Test 1: Basic validation
        success1 = test_basic_validation()
        
        # Test 2: Statistical validation
        success2 = test_statistical_validation()
        
        # Test 3: Batch validation
        success3 = test_batch_validation()
        
        # Test 4: StateBuilder integration
        success4 = test_with_state_builder()
        
        # Test 5: Statistics tracking
        success5 = test_validation_statistics()
        
        if all([success1, success2, success3, success4, success5]):
            logger.info("🎉 All observation validation tests passed!")
            logger.info("✅ Observation validation system is working correctly")
            return True
        else:
            logger.error("❌ Some observation validation tests failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)