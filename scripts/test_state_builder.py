#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the StateBuilder with multi-timeframe data.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_state_builder.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the StateBuilder
try:
    from src.adan_trading_bot.environment.state_builder import StateBuilder
except ImportError as e:
    logger.error(f"Failed to import StateBuilder: {e}")
    sys.exit(1)

def load_data(asset: str = 'BTC', split: str = 'train') -> pd.DataFrame:
    """Load the processed data for a specific asset and split."""
    data_dir = Path('data/final')
    file_path = data_dir / asset / f"{split}.parquet"
    
    if not file_path.exists():
        # Try alternative naming convention
        file_path = data_dir / asset / f"{asset}_{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for {asset} {split}")
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df

def test_state_builder():
    """Test the StateBuilder with sample data."""
    # Configuration
    asset = 'BTC'
    window_size = 30
    # Timeframes ordered from least frequent to most frequent
    timeframes = ['4h', '1h', '5m']
    
    # Features per timeframe (only including what's actually in the data)
    features_per_timeframe = {
        '4h': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update'],
        '1h': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update'],
        '5m': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update']
    }
    
    # Load data
    try:
        df = load_data(asset, 'train')
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        
        # Display column names for debugging
        logger.info("Columns in the data:")
        for col in df.columns:
            logger.info(f"- {col}")
        
        # Filter out any columns that don't match our expected pattern
        valid_columns = []
        for tf in timeframes:
            tf_columns = [col for col in df.columns if col.startswith(f"{tf}_")]
            valid_columns.extend(tf_columns)
            logger.info(f"Found {len(tf_columns)} columns for {tf} timeframe")
        
        # Only keep columns that match our expected features
        df = df[valid_columns + ['target_return']]
        logger.info(f"Filtered data shape: {df.shape}")
        
        # Initialize StateBuilder
        logger.info("Initializing StateBuilder...")
        state_builder = StateBuilder(
            window_size=window_size,
            timeframes=timeframes,
            features_per_timeframe=features_per_timeframe
        )
        
        # Test with a single window of data
        if len(df) >= window_size:
            data_slice = df.iloc[:window_size].copy()
            
            # Build observation
            logger.info("Building observation...")
            observation = state_builder.build_observation(data_slice)
            
            # Log results
            logger.info(f"Observation shape: {observation.shape}")
            logger.info(f"Observation min: {observation.min():.4f}, max: {observation.max():.4f}, mean: {observation.mean():.4f}")
            
            # Print feature names for each timeframe
            feature_names = state_builder.get_feature_names()
            for tf, features in feature_names.items():
                logger.info(f"Features for {tf}: {features}")
            
            return True
        else:
            logger.error(f"Not enough data points. Need at least {window_size}, got {len(df)}")
            return False
            
    except Exception as e:
        logger.error(f"Error in test_state_builder: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting StateBuilder test...")
    success = test_state_builder()
    if success:
        logger.info("StateBuilder test completed successfully!")
    else:
        logger.error("StateBuilder test failed!")
        sys.exit(1)
