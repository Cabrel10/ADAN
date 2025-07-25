#!/usr/bin/env python3
"""
Test script for observation logging functionality.

This script tests the comprehensive observation logging system to ensure
proper logging, metrics collection, and debugging capabilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
import traceback
import json
import time

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.data_processing.observation_validator import ObservationValidator
from adan_trading_bot.common.observation_logger import ObservationLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data for logging tests."""
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
    
    return synthetic_data

def test_basic_logging():
    """Test basic observation logging functionality."""
    logger.info("🔍 Testing basic observation logging...")
    
    # Initialize logger
    obs_logger = ObservationLogger(
        log_dir="logs/test_observations",
        enable_detailed_logging=True,
        enable_metrics_collection=True,
        log_level="INFO"
    )
    
    # Create test observation
    test_obs = np.random.normal(0, 1, (3, 100, 6))
    
    # Test 1: Basic observation logging
    logger.info("📊 Test 1: Basic observation logging...")
    obs_id = obs_logger.log_observation(
        observation=test_obs,
        context={'test': 'basic_logging', 'asset': 'BTC'}
    )
    
    logger.info(f"   Generated observation ID: {obs_id}")
    assert obs_id is not None, "Should generate observation ID"
    logger.info("   ✅ Basic logging test passed")
    
    # Test 2: Logging with validation results
    logger.info("🔍 Test 2: Logging with validation results...")
    validator = ObservationValidator()
    is_valid, validation_results = validator.validate_observation(test_obs)
    
    obs_id_2 = obs_logger.log_observation(
        observation=test_obs,
        context={'test': 'with_validation'},
        validation_results=validation_results
    )
    
    logger.info(f"   Generated observation ID with validation: {obs_id_2}")
    logger.info("   ✅ Validation logging test passed")
    
    # Test 3: Session statistics
    logger.info("📈 Test 3: Session statistics...")
    session_summary = obs_logger.get_session_summary()
    
    logger.info(f"   Session summary: {json.dumps(session_summary, indent=2, default=str)}")
    assert session_summary['total_observations'] >= 2, "Should have logged at least 2 observations"
    logger.info("   ✅ Session statistics test passed")
    
    obs_logger.close()
    return True

def test_batch_logging():
    """Test batch observation logging."""
    logger.info("📦 Testing batch observation logging...")
    
    obs_logger = ObservationLogger(
        log_dir="logs/test_batch_observations",
        enable_detailed_logging=True,
        log_level="INFO"
    )
    
    # Create batch of observations
    batch_size = 5
    batch_obs = np.random.normal(0, 1, (batch_size, 3, 100, 6))
    
    # Log the batch
    batch_context = {'test': 'batch_logging', 'batch_type': 'training'}
    obs_ids = obs_logger.log_batch_observations(
        observations=batch_obs,
        context=batch_context
    )
    
    logger.info(f"   Generated {len(obs_ids)} observation IDs for batch")
    assert len(obs_ids) == batch_size, f"Should generate {batch_size} IDs"
    
    # Check session statistics
    session_summary = obs_logger.get_session_summary()
    assert session_summary['total_observations'] == batch_size, f"Should have {batch_size} observations"
    
    logger.info("   ✅ Batch logging test passed")
    
    obs_logger.close()
    return True

def test_state_builder_integration():
    """Test integration with StateBuilder."""
    logger.info("🔧 Testing StateBuilder integration...")
    
    obs_logger = ObservationLogger(
        log_dir="logs/test_state_builder",
        enable_detailed_logging=True,
        enable_metrics_collection=True,
        log_level="INFO"
    )
    
    # Create test data
    synthetic_data = create_test_data()
    
    # Initialize StateBuilder
    state_builder = StateBuilder(
        window_size=100,
        normalize=True
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
    
    # Generate observation
    current_idx = 150
    multi_obs = state_builder.build_multi_channel_observation(current_idx, synthetic_data)
    
    if multi_obs is not None:
        # Log with StateBuilder context
        data_info = {
            'data_source': 'synthetic',
            'current_idx': current_idx,
            'data_shapes': {tf: df.shape for tf, df in synthetic_data.items()}
        }
        
        obs_id = obs_logger.log_state_builder_output(
            state_builder=state_builder,
            observation=multi_obs,
            current_idx=current_idx,
            data_info=data_info
        )
        
        logger.info(f"   StateBuilder observation logged with ID: {obs_id}")
        
        # Verify the observation was logged correctly
        session_summary = obs_logger.get_session_summary()
        assert session_summary['total_observations'] >= 1, "Should have logged StateBuilder observation"
        
        logger.info("   ✅ StateBuilder integration test passed")
    else:
        logger.error("   ❌ StateBuilder generated None observation")
        return False
    
    obs_logger.close()
    return True

def test_metrics_export():
    """Test metrics export functionality."""
    logger.info("📊 Testing metrics export...")
    
    obs_logger = ObservationLogger(
        log_dir="logs/test_metrics_export",
        enable_metrics_collection=True,
        log_level="INFO"
    )
    
    # Log several observations
    for i in range(3):
        test_obs = np.random.normal(0, 1, (3, 100, 6))
        obs_logger.log_observation(
            observation=test_obs,
            context={'test': 'metrics_export', 'iteration': i}
        )
    
    # Export metrics
    export_file = obs_logger.export_metrics()
    
    logger.info(f"   Metrics exported to: {export_file}")
    
    # Verify export file exists and contains data
    export_path = Path(export_file)
    assert export_path.exists(), "Export file should exist"
    
    with open(export_path, 'r') as f:
        exported_data = json.load(f)
    
    assert 'session_summary' in exported_data, "Should contain session summary"
    assert 'metrics_history' in exported_data, "Should contain metrics history"
    assert len(exported_data['metrics_history']) == 3, "Should have 3 metrics entries"
    
    logger.info("   ✅ Metrics export test passed")
    
    obs_logger.close()
    return True

def test_performance_logging():
    """Test performance aspects of logging."""
    logger.info("⚡ Testing logging performance...")
    
    obs_logger = ObservationLogger(
        log_dir="logs/test_performance",
        enable_detailed_logging=False,  # Disable detailed logging for performance
        enable_metrics_collection=True,
        log_level="WARNING"  # Reduce log verbosity
    )
    
    # Test logging performance with larger observations
    large_obs = np.random.normal(0, 1, (3, 500, 20))  # Larger observation
    
    start_time = time.time()
    
    # Log multiple observations
    num_observations = 10
    for i in range(num_observations):
        obs_logger.log_observation(
            observation=large_obs,
            context={'test': 'performance', 'iteration': i}
        )
    
    total_time = time.time() - start_time
    avg_time_per_obs = total_time / num_observations
    
    logger.info(f"   Logged {num_observations} large observations in {total_time:.3f}s")
    logger.info(f"   Average time per observation: {avg_time_per_obs:.3f}s")
    
    # Check session statistics
    session_summary = obs_logger.get_session_summary()
    logger.info(f"   Average processing time from metrics: {session_summary['average_processing_time_ms']:.2f}ms")
    
    # Performance should be reasonable (less than 100ms per observation)
    assert avg_time_per_obs < 0.1, f"Logging should be fast, got {avg_time_per_obs:.3f}s per observation"
    
    logger.info("   ✅ Performance test passed")
    
    obs_logger.close()
    return True

def test_error_handling():
    """Test error handling in logging."""
    logger.info("🚨 Testing error handling...")
    
    obs_logger = ObservationLogger(
        log_dir="logs/test_error_handling",
        log_level="INFO"
    )
    
    # Test with problematic observations
    
    # Test 1: Observation with NaN values
    nan_obs = np.random.normal(0, 1, (3, 100, 6))
    nan_obs[0, 10:15, 2] = np.nan
    
    obs_id_1 = obs_logger.log_observation(
        observation=nan_obs,
        context={'test': 'error_handling', 'issue': 'nan_values'}
    )
    
    assert obs_id_1 is not None, "Should handle NaN observations gracefully"
    
    # Test 2: Observation with infinite values
    inf_obs = np.random.normal(0, 1, (3, 100, 6))
    inf_obs[1, 20:25, 1] = np.inf
    
    obs_id_2 = obs_logger.log_observation(
        observation=inf_obs,
        context={'test': 'error_handling', 'issue': 'inf_values'}
    )
    
    assert obs_id_2 is not None, "Should handle infinite observations gracefully"
    
    # Check that validation failures were recorded
    session_summary = obs_logger.get_session_summary()
    assert session_summary['validation_failures'] >= 2, "Should record validation failures"
    
    logger.info("   ✅ Error handling test passed")
    
    obs_logger.close()
    return True

def main():
    """Main test function."""
    logger.info("🚀 Starting observation logging tests...")
    
    try:
        # Test 1: Basic logging
        success1 = test_basic_logging()
        
        # Test 2: Batch logging
        success2 = test_batch_logging()
        
        # Test 3: StateBuilder integration
        success3 = test_state_builder_integration()
        
        # Test 4: Metrics export
        success4 = test_metrics_export()
        
        # Test 5: Performance
        success5 = test_performance_logging()
        
        # Test 6: Error handling
        success6 = test_error_handling()
        
        if all([success1, success2, success3, success4, success5, success6]):
            logger.info("🎉 All observation logging tests passed!")
            logger.info("✅ Observation logging system is working correctly")
            return True
        else:
            logger.error("❌ Some observation logging tests failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)