#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the StateBuilder with multi-timeframe data.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_state_builder.log"),
    ],
)
logger = logging.getLogger(__name__)

# Import the StateBuilder
try:
    from src.adan_trading_bot.environment.state_builder import (
        StateBuilder, TimeframeConfig
    )
except ImportError as e:
    logger.error(f"Failed to import StateBuilder: {e}")
    sys.exit(1)


def load_data(asset: str = "BTC", timeframe: str = "5m") -> pd.DataFrame:
    """Load the processed data for a specific asset and timeframe."""
    # Construct absolute path to data directory
    data_dir = Path(project_root) / "data" / "processed" / "indicators"
    file_path = data_dir / asset / f"{timeframe}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(
            f"No data file found for {asset} {timeframe} in {file_path}"
        )

    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df


def test_state_builder():
    """Test the StateBuilder with sample data."""
    # Configuration
    asset = "BTC"
    window_size = 30
    # Timeframes ordered from least frequent to most frequent
    timeframes = ["4h", "1h", "5m"]

    # Features per timeframe with technical indicators
    features_5m = [
        "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "RSI_14", "STOCHk_14_3_3",
        "CCI_20_0.015", "ROC_9", "MFI_14", "EMA_5", "EMA_20",
        "SUPERT_14_2.0", "PSARl_0.02_0.2", "ATRr_14", "BBL_20_2.0",
        "VWAP_D", "OBV",
    ]
    features_1h = [
        "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "RSI_14", "MACD_12_26_9",
        "MACDh_12_26_9", "CCI_20_0.015", "MFI_14", "EMA_50", "EMA_100",
        "SMA_200", "ISA_9", "PSARl_0.02_0.2", "ATRr_14", "BBL_20_2.0",
        "OBV", "VWAP_D",
    ]
    features_4h = [
        "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "RSI_14", "MACD_12_26_9",
        "CCI_20_0.015", "MFI_14", "SMA_200", "EMA_50", "ISA_9",
        "SUPERT_14_3.0", "PSARl_0.02_0.2", "ATRr_14", "BBL_20_2.0",
        "OBV", "VWAP_D",
    ]
    features_per_timeframe = {
        "5m": features_5m,
        "1h": features_1h,
        "4h": features_4h,
    }

    # Load data for each timeframe
    try:
        data = {}
        for tf in timeframes:
            df = load_data(asset, tf)
            logger.info(f"Loaded {tf} data with shape: {df.shape}")
            logger.info(f"Columns in {tf} data: {df.columns.tolist()}")
            data[tf] = df

        # Check if we have data for all timeframes
        if not all(tf in data for tf in timeframes):
            missing = [tf for tf in timeframes if tf not in data]
            raise ValueError(f"Missing data for timeframes: {missing}")

        logger.info(
            f"Successfully loaded data for timeframes: {list(data.keys())}"
        )

        # Initialize StateBuilder with required parameters
        logger.info("Initializing StateBuilder...")
        timeframe_configs = [
            TimeframeConfig(name="4h", features=features_per_timeframe["4h"]),
            TimeframeConfig(name="1h", features=features_per_timeframe["1h"]),
            TimeframeConfig(name="5m", features=features_per_timeframe["5m"]),
        ]

        # Create a mock portfolio manager
        class MockPortfolio:
            def get_state_features(self):
                return np.array([1.0, 0.0, 0.0])  # cash, position, pnl

            def get_feature_size(self):
                return 3

        # Initialize StateBuilder with required parameters
        state_builder = StateBuilder(
            window_size=window_size,
            timeframe_configs=timeframe_configs,
            portfolio_manager=MockPortfolio(),
            assets=["BTC"],
        )
        logger.info("StateBuilder initialized successfully")

        # Test building state for a specific index
        # Skip the first few rows to avoid NaN values
        test_index = window_size + 10

        # Find the minimum length across all timeframes
        min_length = min(len(df) for df in data.values())
        if test_index >= min_length:
            test_index = min_length - 1

        logger.info(f"Building state for index {test_index}...")

        # Build observation for the test index
        observation = {}
        for tf, df in data.items():
            # Get the window of data for this timeframe
            start_idx = max(0, test_index - window_size + 1)
            window_data = df.iloc[start_idx : test_index + 1].copy()

            # Ensure we have enough data points
            if len(window_data) < window_size:
                # Pad with NaN if needed
                padding = pd.DataFrame(
                    np.nan,
                    index=range(window_size - len(window_data)),
                    columns=window_data.columns,
                )
                window_data = pd.concat([padding, window_data], axis=0)

            # Add to observation
            observation[tf] = window_data

        # Build the state
        state = state_builder.build_observation(observation)

        # Log the state structure
        logger.info("State structure:")
        log_msg = (
            f"Overall state shape: {state.shape} | "
            f"dtype={state.dtype} | min={state.min():.4f} | "
            f"max={state.max():.4f} | mean={state.mean():.4f} | "
            f"isnan={np.isnan(state).any()}"
        )
        logger.info(log_msg)

        # Verify the state structure
        assert isinstance(state, np.ndarray), "State should be a numpy array"
        assert not np.isnan(state).any(), "NaN values in the overall state"

        # Verify the state structure
        shape_3d = state_builder.get_observation_shape()
        n_tf, win_size, max_feat = shape_3d
        expected_market_size = n_tf * win_size * max_feat
        portfolio_size = state_builder.portfolio.get_feature_size()
        expected_total_size = expected_market_size + portfolio_size

        logger.info(f"Expected 3D shape: {shape_3d}")
        logger.info(f"Expected market features size: {expected_market_size}")
        logger.info(f"Portfolio features size: {portfolio_size}")
        logger.info(f"Expected total flattened size: {expected_total_size}")

        assert state.shape[0] == expected_total_size, (
            f"Incorrect total feature count. Expected {expected_total_size}, "
            f"got {state.shape[0]}"
        )

        logger.info("✅ StateBuilder test passed!")
        return True

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
