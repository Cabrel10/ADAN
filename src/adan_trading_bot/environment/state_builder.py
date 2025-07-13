#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
State builder for the ADAN trading environment.

This module is responsible for constructing the 3D observation state that is
fed to the reinforcement learning agent at each step. The state combines data
from multiple timeframes into a structured format that preserves the temporal
and feature dimensions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe in the state representation."""
    name: str  # e.g., '5m', '1h', '4h'
    features: List[str]  # List of feature names for this timeframe
    weight: float = 1.0  # Optional weight for this timeframe

class StateBuilder:
    """
    Constructs the 3D observation state for the RL agent.

    The state is a 3D tensor with dimensions:
    - Channel: Timeframe (e.g., 5m, 1h, 4h)
    - Time: Historical time steps (window_size)
    - Features: Market data features for each timeframe
    """
    
    def __init__(self, 
                 window_size: int, 
                 timeframes: List[str],
                 features_per_timeframe: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the StateBuilder with multi-timeframe support.

        Args:
            window_size: Number of time steps to include in the observation window.
            timeframes: List of timeframes to include (e.g., ['5m', '1h', '4h']).
                       Must be ordered from highest to lowest frequency.
            features_per_timeframe: Optional dictionary mapping timeframes to their features.
                                  If None, default features will be used.
        """
        self.window_size = window_size
        self.timeframes = timeframes
        self.features_per_timeframe = features_per_timeframe or {}
        
        # Validate timeframes are ordered from highest to lowest frequency
        self._validate_timeframe_order()
        
        # Set default features for any timeframes not specified
        self._set_default_features()
        
        # Precompute feature indices for faster access
        self._setup_feature_indices()
        
        logger.info(f"StateBuilder initialized with window_size={window_size}, "
                  f"timeframes={timeframes}, features_per_timeframe={self.features_per_timeframe}")
    
    def _validate_timeframe_order(self) -> None:
        """Ensure timeframes are ordered from highest to lowest frequency."""
        # Define the canonical order of timeframes from highest to lowest frequency
        canonical_order = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
        
        # Get the indices of our timeframes in the canonical order
        indices = []
        for tf in self.timeframes:
            try:
                indices.append(canonical_order.index(tf))
            except ValueError:
                logger.warning(f"Timeframe {tf} not in canonical order, may cause unexpected behavior")
                indices.append(-1)
        
        # Check if the indices are in descending order (higher frequency first)
        if not all(i > j for i, j in zip(indices, indices[1:])):
            logger.warning(f"Timeframes should be ordered from highest to lowest frequency. Got: {self.timeframes}")
    
    def _set_default_features(self) -> None:
        """Set default features for each timeframe if not specified."""
        # Common OHLCV features
        ohlcv_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Additional derived features that might be available
        additional_features = [
            'returns', 'log_returns', 'volatility', 'rsi', 'macd', 'macd_signal',
            'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv'
        ]
        
        for tf in self.timeframes:
            if tf not in self.features_per_timeframe:
                # Start with OHLCV features
                self.features_per_timeframe[tf] = ohlcv_features.copy()
                
                # Add any additional features that might be available
                # These will be filtered out later if not present in the data
                self.features_per_timeframe[tf].extend(additional_features)
                
                logger.debug(f"Using default features for {tf}: {self.features_per_timeframe[tf]}")
    
    def set_features_for_timeframe(self, timeframe: str, features: List[str]) -> None:
        """
        Set the features to include for a specific timeframe.
        
        Args:
            timeframe: The timeframe to set features for (e.g., '5m').
            features: List of feature names to include.
        """
        if timeframe not in self.timeframes:
            logger.warning(f"Timeframe {timeframe} not in configured timeframes: {self.timeframes}")
            return
            
        self.features_per_timeframe[timeframe] = features
        logger.info(f"Set {len(features)} features for timeframe {timeframe}")
    
    def get_observation_shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of the observation tensor.
        
        Returns:
            Tuple of (num_timeframes, window_size, num_features)
            where num_features is the maximum number of features across all timeframes
        """
        return (len(self.timeframes), self.window_size, self.max_features)
    
    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Apply normalization to the observation tensor.
        
        This is a placeholder that can be overridden by subclasses to implement
        specific normalization strategies (e.g., z-score, min-max, etc.).
        
        Args:
            observation: The 3D observation tensor to normalize
            
        Returns:
            Normalized observation tensor
        """
        # Default implementation: simple min-max scaling to [0, 1] per feature
        # Skip this if the values are already normalized
        if np.max(np.abs(observation)) > 1.0:
            # Avoid division by zero
            max_vals = np.max(np.abs(observation), axis=1, keepdims=True)
            max_vals[max_vals < 1e-8] = 1.0  # Avoid division by zero
            
            observation = observation / max_vals
            
        return observation
    
    def _setup_feature_indices(self) -> None:
        """Precompute feature indices for faster access during observation building."""
        self.feature_indices = {}
        self.max_features = 0
        
        for tf in self.timeframes:
            self.feature_indices[tf] = {}
            for i, feature in enumerate(self.features_per_timeframe[tf]):
                self.feature_indices[tf][feature] = i
            
            # Track the maximum number of features across all timeframes
            num_features = len(self.features_per_timeframe[tf])
            self.max_features = max(self.max_features, num_features)
    
    def build_observation(self, data_slice: pd.DataFrame) -> np.ndarray:
        """
        Build a 3D observation tensor from the given data slice.
        
        The observation tensor has shape (num_timeframes, window_size, num_features)
        where:
        - num_timeframes: Number of timeframes (e.g., 3 for 5m, 1h, 4h)
        - window_size: Number of time steps in the lookback window
        - num_features: Number of features per timeframe
        
        Args:
            data_slice: DataFrame containing the window of data to build the observation from.
                       Must have columns prefixed with timeframe (e.g., '5m_open', '1h_close').
        
        Returns:
            A 3D numpy array of shape (num_timeframes, window_size, num_features)
            
        Raises:
            ValueError: If the input data doesn't match expected format or is empty
        """
        if data_slice is None or data_slice.empty:
            raise ValueError("Input data slice is empty")
            
        if len(data_slice) != self.window_size:
            logger.warning(f"Expected window_size={self.window_size}, got {len(data_slice)}. "
                         f"This may cause issues with the observation shape.")
            
        # Initialize the 3D observation array
        num_timeframes = len(self.timeframes)
        observation = np.zeros((num_timeframes, self.window_size, self.max_features), 
                             dtype=np.float32)
        
        # Process each timeframe
        for tf_idx, tf in enumerate(self.timeframes):
            # Get the features for this timeframe
            tf_features = self.features_per_timeframe[tf]
            
            # For each feature, try to find it in the dataframe
            for feat_idx, feature in enumerate(tf_features):
                col_name = f"{tf}_{feature}"
                
                if col_name in data_slice.columns:
                    # Get the values for this feature
                    values = data_slice[col_name].values
                    
                    # Ensure we have the right number of values
                    if len(values) < self.window_size:
                        # Pad with zeros if we don't have enough values
                        padded = np.zeros(self.window_size, dtype=np.float32)
                        padded[-len(values):] = values
                        values = padded
                    elif len(values) > self.window_size:
                        # Truncate if we have too many values
                        values = values[-self.window_size:]
                    
                    observation[tf_idx, :, feat_idx] = values
                else:
                    # Feature not found, leave as zeros
                    logger.debug(f"Feature {col_name} not found in data, using zeros")
        
        # Apply normalization if needed (can be overridden by subclasses)
        observation = self._normalize_observation(observation)
        
        return observation
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get the feature names for each timeframe.
        
        Returns:
            Dictionary mapping timeframes to their feature lists.
        """
        return self.features_per_timeframe.copy()