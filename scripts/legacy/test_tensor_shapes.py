#!/usr/bin/env python3
"""
Test script to verify tensor shapes according to design specifications.

This script tests that the StateBuilder generates observations with the correct
3D shape (3, window_size, nb_features) as specified in the design document.
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
# from adan_trading_bot.data_processing.chunked_loader import ChunkedDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_observation_shape():
    """Test that observations have the correct shape according to design specifications."""
    logger.info("🔍 Testing tensor shapes according to design specifications...")
    
    # Design specifications
    timeframes = ['5m', '1h', '4h']
    window_size = 100
    
    # Features configuration according to design
    features_config = {
        '5m': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update'],
        '1h': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update'],
        '4h': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update']
    }
    
    # Initialize StateBuilder
    state_builder = StateBuilder(
        features_config=features_config,
        window_size=window_size,
        include_portfolio_state=True,
        normalize=True
    )
    
    # Test 1: Verify expected observation shape
    logger.info("📏 Test 1: Verifying expected observation shape...")
    expected_shape = state_builder.get_observation_shape()
    max_features = max(len(features) for features in features_config.values())
    design_expected_shape = (len(timeframes), window_size, max_features)
    
    logger.info(f"   Expected shape from StateBuilder: {expected_shape}")
    logger.info(f"   Expected shape from design: {design_expected_shape}")
    
    assert expected_shape == design_expected_shape, f"Shape mismatch: {expected_shape} vs {design_expected_shape}"
    logger.info("   ✅ Shape specification matches design")
    
    # Test 2: Test with synthetic data
    logger.info("📊 Test 2: Testing with synthetic data...")
    
    # Create synthetic data for testing
    synthetic_data = {}
    for tf in timeframes:
        # Create synthetic DataFrame with required features
        n_samples = window_size + 50  # Extra samples for testing
        data = {
            'open': np.random.uniform(100, 200, n_samples),
            'high': np.random.uniform(150, 250, n_samples),
            'low': np.random.uniform(50, 150, n_samples),
            'close': np.random.uniform(100, 200, n_samples),
            'volume': np.random.uniform(1000, 10000, n_samples),
            'minutes_since_update': np.random.randint(0, 60, n_samples)
        }
        
        # Ensure high >= close >= low and high >= open >= low
        for i in range(n_samples):
            data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
            data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
        
        synthetic_data[tf] = pd.DataFrame(data)
        logger.info(f"   Created synthetic data for {tf}: {synthetic_data[tf].shape}")
    
    # Fit scalers on synthetic data
    state_builder.fit_scalers(synthetic_data)

    # --- Dimension Validation ---
    logger.info("📏 Test 2.5: Validating state dimension...")
    try:
        state_builder.validate_dimension(synthetic_data)
        logger.info("   ✅ State dimension validation successful.")
    except ValueError as e:
        logger.error(f"   ❌ State dimension validation failed: {e}")
        raise e
    # --- End Validation ---
    
    # Test observation building
    current_idx = window_size + 10  # Use an index with enough history
    
    # Test individual timeframe observations
    logger.info("🔧 Test 3: Testing individual timeframe observations...")
    observations = state_builder.build_observation(current_idx, synthetic_data)
    
    for tf in timeframes:
        if tf in observations:
            obs_shape = observations[tf].shape
            expected_tf_shape = (window_size, len(features_config[tf]))
            logger.info(f"   {tf} observation shape: {obs_shape}, expected: {expected_tf_shape}")
            assert obs_shape == expected_tf_shape, f"Shape mismatch for {tf}: {obs_shape} vs {expected_tf_shape}"
        else:
            logger.warning(f"   No observation generated for {tf}")
    
    logger.info("   ✅ Individual timeframe observations have correct shapes")
    
    # Test multi-channel observation (main test)
    logger.info("🎯 Test 4: Testing multi-channel observation (MAIN TEST)...")
    multi_obs = state_builder.build_multi_channel_observation(current_idx, synthetic_data)
    
    if multi_obs is not None:
        actual_shape = multi_obs.shape
        logger.info(f"   Multi-channel observation shape: {actual_shape}")
        logger.info(f"   Expected shape: {design_expected_shape}")
        
        # Verify shape matches design specification
        assert actual_shape == design_expected_shape, f"Multi-channel shape mismatch: {actual_shape} vs {design_expected_shape}"
        
        # Verify data quality
        assert np.isfinite(multi_obs).all(), "Multi-channel observation contains NaN or infinite values"
        
        # Check normalization (values should be reasonable after scaling)
        max_abs_value = np.abs(multi_obs).max()
        logger.info(f"   Maximum absolute value after normalization: {max_abs_value:.4f}")
        
        if max_abs_value > 10:
            logger.warning(f"   Values seem unnormalized: max absolute value is {max_abs_value}")
        
        logger.info("   ✅ Multi-channel observation has correct 3D shape!")
        
    else:
        logger.error("   ❌ Multi-channel observation is None")
        return False
    
    # Test 5: Validation function
    logger.info("✅ Test 5: Testing observation validation...")
    is_valid = state_builder.validate_observation(multi_obs)
    assert is_valid, "Observation validation failed"
    logger.info("   ✅ Observation validation passed")
    
    return True

def test_with_real_data():
    """Test tensor shapes with real data if available."""
    logger.info("📈 Testing with real data...")
    
    # Check if real data is available
    data_dir = Path(__file__).parent.parent / 'data' / 'final'
    if not data_dir.exists():
        logger.info("   Real data directory not found, skipping real data test")
        return True
    
    # Look for available assets
    available_assets = []
    for asset_dir in data_dir.iterdir():
        if asset_dir.is_dir() and (asset_dir / 'train.parquet').exists():
            available_assets.append(asset_dir.name)
    
    if not available_assets:
        logger.info("   No real data files found, skipping real data test")
        return True
    
    # Use the first available asset
    test_asset = available_assets[0]
    logger.info(f"   Testing with real data for asset: {test_asset}")
    
    try:
        # Load real data directly using pandas
        asset_dir = data_dir / test_asset
        train_file = asset_dir / 'train.parquet'
        
        if not train_file.exists():
            logger.info("   Train data file not found, skipping real data test")
            return True
        
        # Load the parquet file
        df = pd.read_parquet(train_file)
        logger.info(f"   Loaded real data with shape: {df.shape}")
        logger.info(f"   Available columns: {list(df.columns)}")
        
        # Create timeframe data structure with correct column mapping
        timeframes = ['5m', '1h', '4h']
        chunk_data = {}
        
        # Map the real data columns to the expected format
        for tf in timeframes:
            tf_columns = [col for col in df.columns if col.startswith(f'{tf}_')]
            if tf_columns:
                # Create a DataFrame with the expected column names
                tf_data = df[tf_columns].copy()
                # Rename columns to remove the timeframe prefix
                tf_data.columns = [col.replace(f'{tf}_', '') for col in tf_data.columns]
                chunk_data[tf] = tf_data
            else:
                # Fallback: create synthetic data for this timeframe
                n_samples = len(df)
                synthetic_tf_data = {
                    'open': np.random.uniform(100, 200, n_samples),
                    'high': np.random.uniform(150, 250, n_samples),
                    'low': np.random.uniform(50, 150, n_samples),
                    'close': np.random.uniform(100, 200, n_samples),
                    'volume': np.random.uniform(1000, 10000, n_samples),
                    'minutes_since_update': np.random.randint(0, 60, n_samples)
                }
                chunk_data[tf] = pd.DataFrame(synthetic_tf_data)
        
        # Initialize StateBuilder
        state_builder = StateBuilder(
            window_size=100,
            normalize=True
        )
        
        # Fit scalers on real data
        state_builder.fit_scalers(chunk_data)

        # --- Dimension Validation with Real Data ---
        logger.info("📏 Validating state dimension with real data...")
        try:
            state_builder.validate_dimension(chunk_data)
            logger.info("   ✅ Real data state dimension validation successful.")
        except ValueError as e:
            logger.error(f"   ❌ Real data state dimension validation failed: {e}")
            raise e
        # --- End Validation ---
        
        # Test observation building with real data
        current_idx = min(150, len(df) - 1)  # Use an index with enough history
        
        # Build multi-channel observation
        multi_obs = state_builder.build_multi_channel_observation(current_idx, chunk_data)
        
        if multi_obs is not None:
            logger.info(f"   Real data observation shape: {multi_obs.shape}")
            
            # Validate the observation
            is_valid = state_builder.validate_observation(multi_obs)
            assert is_valid, "Real data observation validation failed"
            
            logger.info("   ✅ Real data test passed!")
        else:
            logger.warning("   Real data observation is None")
            
        return True
        
    except Exception as e:
        logger.error(f"   Error testing with real data: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting tensor shape verification tests...")
    
    try:
        # Test 1: Synthetic data test
        success1 = test_observation_shape()
        
        # Test 2: Real data test (if available)
        success2 = test_with_real_data()
        
        if success1 and success2:
            logger.info("🎉 All tensor shape tests passed!")
            logger.info("✅ StateBuilder generates observations with correct 3D shape according to design specifications")
            return True
        else:
            logger.error("❌ Some tensor shape tests failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)