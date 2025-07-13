"""
State builder for creating multi-timeframe observations for the RL agent.

This module provides the StateBuilder class which transforms raw market data
into a structured observation space suitable for reinforcement learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateBuilder:
    """
    Builds state representations from multi-timeframe market data.
    
    This class handles the transformation of raw market data into a structured
    observation space that can be used by reinforcement learning agents.
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 timeframes: List[str] = None,
                 feature_columns: List[str] = None,
                 normalize: bool = True,
                 scaler_path: Optional[str] = None):
        """
        Initialize the StateBuilder.
        
        Args:
            window_size: Number of time steps to include in each observation
            timeframes: List of timeframes to include (e.g., ['1m', '1h', '3h'])
            feature_columns: List of feature column names to include
            normalize: Whether to normalize the data
            scaler_path: Path to save/load the scaler
        """
        self.window_size = window_size
        self.timeframes = timeframes or ['1m', '1h', '3h']
        self.feature_columns = feature_columns or [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'boll_upper', 'boll_middle', 'boll_lower'
        ]
        self.normalize = normalize
        self.scaler_path = scaler_path
        self.scalers = {}
        self.feature_indices = {}
        
        # Initialize scalers for each timeframe
        self._init_scalers()
    
    def _init_scalers(self) -> None:
        """Initialize scalers for each timeframe."""
        for tf in self.timeframes:
            if self.normalize:
                self.scalers[tf] = StandardScaler()
            else:
                self.scalers[tf] = None
    
    def fit_scalers(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit scalers on the provided data.
        
        Args:
            data: Dictionary mapping timeframes to DataFrames
        """
        if not self.normalize:
            return
            
        logger.info("Fitting scalers on provided data...")
        
        # Combine all data for fitting
        all_data = []
        for tf, df in data.items():
            if tf not in self.timeframes:
                continue
                
            # Select only the feature columns that exist in the DataFrame
            columns = [col for col in self.feature_columns if col in df.columns]
            if not columns:
                logger.warning(f"No matching feature columns found for timeframe {tf}")
                continue
                
            # Store the indices of the selected features
            self.feature_indices[tf] = [self.feature_columns.index(col) for col in columns]
            
            # Add to the combined dataset for fitting
            all_data.append(df[columns].values)
        
        if not all_data:
            logger.warning("No data available for fitting scalers")
            return
            
        # Combine all data and fit the scaler
        combined_data = np.vstack(all_data)
        
        # Handle potential infinite or NaN values
        if not np.isfinite(combined_data).all():
            logger.warning("Non-finite values found in data. Replacing with zeros.")
            combined_data = np.nan_to_num(combined_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit the scaler on the combined data
        for tf in self.timeframes:
            if self.scalers[tf] is not None:
                self.scalers[tf].fit(combined_data)
                logger.info(f"Fitted scaler for timeframe {tf} on {len(combined_data)} samples")
        
        # Save the scaler if a path is provided
        if self.scaler_path:
            self.save_scalers()
    
    def save_scalers(self) -> None:
        """Save the fitted scalers to disk."""
        if not self.scaler_path:
            return
            
        scaler_dir = Path(self.scaler_path).parent
        scaler_dir.mkdir(parents=True, exist_ok=True)
        
        for tf, scaler in self.scalers.items():
            if scaler is not None:
                scaler_file = scaler_dir / f"scaler_{tf}.joblib"
                joblib.dump(scaler, scaler_file)
                logger.info(f"Saved scaler for {tf} to {scaler_file}")
    
    def load_scalers(self) -> bool:
        """
        Load fitted scalers from disk.
        
        Returns:
            bool: True if all scalers were loaded successfully, False otherwise
        """
        if not self.scaler_path:
            return False
            
        all_loaded = True
        scaler_dir = Path(self.scaler_path).parent
        
        for tf in self.timeframes:
            scaler_file = scaler_dir / f"scaler_{tf}.joblib"
            if scaler_file.exists():
                try:
                    self.scalers[tf] = joblib.load(scaler_file)
                    logger.info(f"Loaded scaler for {tf} from {scaler_file}")
                except Exception as e:
                    logger.error(f"Error loading scaler for {tf}: {e}")
                    all_loaded = False
            else:
                logger.warning(f"Scaler file not found for {tf}: {scaler_file}")
                all_loaded = False
        
        return all_loaded
    
    def build_observation(self, 
                         current_idx: int, 
                         data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Build a multi-timeframe observation.
        
        Args:
            current_idx: Current index in the data
            data: Dictionary mapping timeframes to DataFrames
            
        Returns:
            Dictionary mapping timeframes to observation arrays
        """
        observations = {}
        
        for tf, df in data.items():
            if tf not in self.timeframes:
                continue
                
            # Get the window of data
            start_idx = max(0, current_idx - self.window_size + 1)
            df_window = df.iloc[start_idx:current_idx+1].copy()
            
            # Select only the feature columns that exist in the DataFrame
            columns = [col for col in self.feature_columns if col in df_window.columns]
            if not columns:
                logger.warning(f"No matching feature columns found for timeframe {tf}")
                continue
                
            # Get the feature values
            values = df_window[columns].values
            
            # Pad with zeros if we don't have enough history
            if len(values) < self.window_size:
                padding = np.zeros((self.window_size - len(values), len(columns)))
                values = np.vstack([padding, values])
            
            # Apply normalization if enabled
            if self.normalize and self.scalers.get(tf) is not None:
                try:
                    # Reshape for scaler (n_samples, n_features) -> (n_samples * n_features, 1)
                    original_shape = values.shape
                    values_flat = values.reshape(-1, 1)
                    
                    # Scale the data
                    values_scaled = self.scalers[tf].transform(values_flat)
                    
                    # Reshape back to original shape
                    values = values_scaled.reshape(original_shape)
                except Exception as e:
                    logger.error(f"Error normalizing data for {tf}: {e}")
            
            # Add to observations
            observations[tf] = values
        
        return observations
    
    def build_multi_channel_observation(self, 
                                      current_idx: int, 
                                      data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Build a multi-channel observation with all timeframes.
        
        Args:
            current_idx: Current index in the data
            data: Dictionary mapping timeframes to DataFrames
            
        Returns:
            3D numpy array of shape (n_timeframes, window_size, n_features)
        """
        # Get observations for each timeframe
        observations = self.build_observation(current_idx, data)
        
        if not observations:
            return None
            
        # Find the maximum number of features across all timeframes
        max_features = max(obs.shape[1] for obs in observations.values())
        
        # Initialize the output array
        n_timeframes = len(self.timeframes)
        output = np.zeros((n_timeframes, self.window_size, max_features))
        
        # Fill the output array
        for i, tf in enumerate(self.timeframes):
            if tf in observations:
                obs = observations[tf]
                # Center the features if they have fewer columns than max_features
                padding = max_features - obs.shape[1]
                if padding > 0:
                    left_pad = padding // 2
                    right_pad = padding - left_pad
                    obs = np.pad(obs, ((0, 0), (left_pad, right_pad)), 
                               mode='constant', constant_values=0)
                output[i, :, :] = obs
        
        return output
