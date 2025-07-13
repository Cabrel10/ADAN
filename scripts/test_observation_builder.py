#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the observation builder with multi-timeframe data.
This script loads sample data, builds observations, and verifies their structure.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch as th
from pathlib import Path
import logging
import yaml

# Add src to PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from src.adan_trading_bot.agent.feature_extractors import CustomCNNFeatureExtractor
from src.adan_trading_bot.environment.multi_asset_env import AdanTradingEnv
from src.adan_trading_bot.data_processing.data_loader import ComprehensiveDataLoader
from src.adan_trading_bot.data_processing.state_builder import StateBuilder
from src.adan_trading_bot.common.utils import get_logger

# Configure logger
logger = get_logger()
logger.setLevel(logging.INFO)

# Configuration
CONFIG_PATH = "config/main_config.yaml"
ASSET = "BTCUSDT"
WINDOW_SIZE = 100  # Number of time steps in the observation window

def load_config(config_path):
    """Load configuration from YAML file, handling includes and path resolution."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent # Assuming project root is one level above scripts

    with open(config_path, 'r') as f:
        main_config = yaml.safe_load(f)

    # Handle includes
    if 'imports' in main_config:
        for import_file in main_config['imports']:
            imported_config_path = Path(config_path).parent / import_file
            with open(imported_config_path, 'r') as f_imported:
                imported_config = yaml.safe_load(f_imported)
                # Merge imported config into main config
                # This is a shallow merge. For deeper merges, a recursive function is needed.
                main_config.update(imported_config)
        del main_config['imports'] # Remove the imports key after processing

    # Resolve paths relative to the project root
    if 'paths' in main_config:
        for key, value in main_config['paths'].items():
            if isinstance(value, str):
                # Replace placeholders like ${paths.base_dir}
                for path_key, path_value in main_config['paths'].items():
                    if f"${{paths.{path_key}}}" in value:
                        value = value.replace(f"${{paths.{path_key}}}", path_value)
                
                # Resolve relative paths against the project root
                if not Path(value).is_absolute():
                    main_config['paths'][key] = str(project_root / value)
                else:
                    main_config['paths'][key] = value
                
                # Expand environment variables (e.g., ${HOME})
                main_config['paths'][key] = os.path.expandvars(main_config['paths'][key])

    return main_config

def test_observation_builder():
    """Test the observation builder with multi-timeframe data."""
    # Load configuration
    config = load_config(CONFIG_PATH)
    
    # Initialize data loader
    data_loader = ComprehensiveDataLoader(
        data_config=config,
        base_data_dir=config['paths']['raw_data_dir']
    )
    data_loader.load_and_merge_data()
    
    # Get the first chunk of data
    chunk_data = data_loader.get_next_chunk()
    if chunk_data is None or chunk_data.empty:
        logger.error(f"Failed to get initial chunk of data for {ASSET}")
        return False
    
    logger.info(f"Loaded data for {ASSET}, shape: {chunk_data.shape}")
    
    # Initialize state builder
    window_size = config['reward_shaping']['chunk_size'] # Assuming window_size is chunk_size for now
    timeframes = config['feature_engineering']['timeframes']
    feature_columns = config['feature_engineering']['technical_indicators']
    scaler_path = str(Path(config['paths']['models_dir']) / "scaler.joblib")

    state_builder = StateBuilder(
        window_size=window_size,
        timeframes=timeframes,
        feature_columns=feature_columns,
        normalize=True,
        scaler_path=scaler_path
    )
    
    # Initialize the feature extractor
    # Create a dummy observation space to get the correct shapes
    obs_space = {
        'image_features': th.zeros((3, WINDOW_SIZE, len(config['features']['technical_indicators']))),
        'vector_features': th.zeros(5)  # Example size, adjust as needed
    }
    
    feature_extractor = CustomCNNFeatureExtractor(obs_space)
    
    # Process each time step and verify observations
    for i in range(WINDOW_SIZE, len(chunk_data)):
        # Get window of data
        window_data = chunk_data.iloc[i-WINDOW_SIZE:i]
        
        # Build observation
        observation = state_builder.build_observation(window_data)
        
        # Verify observation structure
        if not isinstance(observation, dict):
            logger.error(f"Observation should be a dict, got {type(observation)}")
            return False
            
        if 'image_features' not in observation or 'vector_features' not in observation:
            logger.error("Observation missing required keys: 'image_features' and 'vector_features'")
            return False
            
        # Verify image features shape
        img_features = observation['image_features']
        expected_shape = (3, WINDOW_SIZE, len(config['features']['technical_indicators']))
        if img_features.shape != expected_shape:
            logger.error(f"Unexpected image features shape: {img_features.shape}, expected {expected_shape}")
            return False
        
        # Verify vector features
        vec_features = observation['vector_features']
        if not isinstance(vec_features, np.ndarray) or len(vec_features.shape) != 1:
            logger.error(f"Vector features should be 1D numpy array, got {type(vec_features)}")
            return False
        
        # Test the feature extractor
        try:
            # Convert to PyTorch tensors
            img_tensor = th.FloatTensor(img_features).unsqueeze(0)  # Add batch dimension
            vec_tensor = th.FloatTensor(vec_features).unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            features = feature_extractor({
                'image_features': img_tensor,
                'vector_features': vec_tensor
            })
            
            logger.info(f"Successfully processed observation at index {i}")
            logger.info(f"Feature extractor output shape: {features.shape}")
            
            # If we get here, everything worked for this observation
            return True
            
        except Exception as e:
            logger.error(f"Error processing observation at index {i}: {str(e)}")
            logger.exception("Full traceback:")
            return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting observation builder test...")
    success = test_observation_builder()
    
    if success:
        logger.info("Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Test failed!")
        sys.exit(1)
