"""
State builder for creating multi-timeframe observations for the RL agent.

This module provides the StateBuilder class which transforms raw market data
into a structured observation space suitable for reinforcement learning.
"""

import gc
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

class TimeframeConfig:
    """
    Configuration class for timeframe-specific settings.
    
    This class encapsulates the configuration for a specific timeframe,
    including its features and any other relevant settings.
    """
    def __init__(self, 
                 timeframe: str,
                 features: List[str],
                 window_size: int = 100,
                 normalize: bool = True):
        """
        Initialize timeframe configuration.
        
        Args:
            timeframe: The timeframe identifier (e.g., '5m', '1h')
            features: List of feature names for this timeframe
            window_size: Number of time steps to include
            normalize: Whether to normalize the data
        """
        self.timeframe = timeframe
        self.features = features
        self.window_size = window_size
        self.normalize = normalize

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'timeframe': self.timeframe,
            'features': self.features,
            'window_size': self.window_size,
            'normalize': self.normalize
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TimeframeConfig':
        """Create configuration from dictionary."""
        return cls(
            timeframe=config_dict['timeframe'],
            features=config_dict['features'],
            window_size=config_dict.get('window_size', 100),
            normalize=config_dict.get('normalize', True)
        )

class StateBuilder:
    """
    Builds state representations from multi-timeframe market data.
    
    This class handles the transformation of raw market data into a structured
    observation space that can be used by reinforcement learning agents.
    """
    
    def __init__(self,
                 features_config: Dict[str, List[str]] = None,
                 window_size: int = 20,  # Correspond à la configuration dans config.yaml
                 include_portfolio_state: bool = True,
                 normalize: bool = True,
                 scaler_path: Optional[str] = None,
                 adaptive_window: bool = True,
                 min_window_size: int = 10,  # 50% de la taille de fenêtre par défaut
                 max_window_size: int = 30,  # 150% de la taille de fenêtre par défaut
                 memory_config: Optional[Dict[str, Any]] = None,  # Configuration de mémoire
                 target_observation_size: Optional[int] = None):
        """
        Initialize the StateBuilder according to design specifications.
        
        Args:
            features_config: Dictionary mapping timeframes to their feature lists
            window_size: Base number of time steps to include in each observation
            include_portfolio_state: Whether to include portfolio state in observations
            normalize: Whether to normalize the data
            scaler_path: Path to save/load the scaler
            adaptive_window: Whether to use adaptive window sizing based on volatility
            min_window_size: Minimum window size for adaptive mode
            max_window_size: Maximum window size for adaptive mode
            memory_config: Configuration for memory optimizations
        """
        # Configuration initiale
        # Utiliser la configuration exacte de config.yaml
        if features_config is None:
            features_config = {
                "5m": ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "RSI_14", "STOCHk_14_3_3", "STOCHd_14_3_3", 
                        "CCI_20_0.015", "ROC_9", "MFI_14", "EMA_5", "EMA_20", "SUPERTREND_14_2.0", "PSAR_0.02_0.2"],
                "1h": ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "RSI_14", "MACD_12_26_9", "MACD_HIST_12_26_9", 
                        "CCI_20_0.015", "MFI_14", "EMA_50", "EMA_100", "SMA_200", "ICHIMOKU_9_26_52", "PSAR_0.02_0.2"],
                "4h": ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "RSI_14", "MACD_12_26_9", "CCI_20_0.015", 
                        "MFI_14", "EMA_50", "SMA_200", "ICHIMOKU_9_26_52", "SUPERTREND_14_3.0", "PSAR_0.02_0.2"]
            }
        self.features_config = features_config
        # Ne garder que les timeframes qui ont des features définies
        self.timeframes = [tf for tf in ["5m", "1h", "4h"] if tf in self.features_config]
        if not self.timeframes:
            raise ValueError("Aucun timeframe valide trouvé dans la configuration des fonctionnalités")
            
        self.nb_features_per_tf = {tf: len(features) for tf, features in self.features_config.items() 
                                 if tf in self.timeframes}
        
        # Configuration de mémoire
        self.memory_config = memory_config or {
            'aggressive_cleanup': True,
            'force_gc': True,
            'memory_monitoring': True,
            'memory_warning_threshold_mb': 5600,
            'memory_critical_threshold_mb': 6300,
            'disable_caching': True
        }
        
        # Métriques de performance
        self.performance_metrics = {
            'gc_collections': 0,
            'memory_peak_mb': 0,
            'errors_count': 0,
            'warnings_count': 0
        }
        
        # Mémoire initiale
        self.initial_memory_mb = 0
        self.memory_peak_mb = 0
        
        # Initialiser les métriques après la configuration
        self._initialize_memory_metrics()
        
        # Configuration de la taille de fenêtre
        self.base_window_size = window_size
        # Configuration de la fenêtre fixe
        self.window_size = 100  # Taille fixe de la fenêtre
        self.include_portfolio_state = include_portfolio_state
        self.normalize = normalize
        
        # Maximum de features défini dans la config
        # Déterminer le nombre maximum de features parmi tous les timeframes
        self.max_features = max(len(features) for features in self.features_config.values()) if self.features_config else 0
        
        # Forme dynamique : (nombre de timeframes, fenêtre, max_features)
        self.observation_shape = (len(self.timeframes), self.window_size, self.max_features)  
        
        # Configuration adaptative
        self.adaptive_window = adaptive_window
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.volatility_history = []
        self.volatility_window = 20
        self.timeframe_weights = {tf: 1.0 for tf in self.timeframes} # Initialisation des poids

        # Configuration des scalers
        self.scaler_path = scaler_path
        self.scalers = {tf: None for tf in self.timeframes}
        self.feature_indices = {}

        # Initialisation des scalers
        self._init_scalers()

        # The observation_shape for StateBuilder will be the 3D shape
        self.total_flattened_observation_size = self.observation_shape[0] * self.observation_shape[1] * self.observation_shape[2]
        if self.include_portfolio_state:
            self.total_flattened_observation_size += 17

        logger.info(f"StateBuilder initialized. Target flattened observation size: {self.total_flattened_observation_size}")
        logger.info(f"Features per timeframe: {self.nb_features_per_tf}")
        logger.info(f"StateBuilder initialized with base_window_size={window_size}, "
                   f"adaptive_window={adaptive_window}, "
                   f"timeframes={self.timeframes}, "
                   f"features_per_timeframe={self.nb_features_per_tf}")

    def _initialize_memory_metrics(self):
        """
        Initialize memory metrics after configuration.
        """
        try:
            # Get initial memory usage
            self.initial_memory_mb = self._get_memory_usage_mb()
            self.memory_peak_mb = self.initial_memory_mb
            
            # Update performance metrics
            self._update_performance_metrics('memory_peak_mb', self.initial_memory_mb)
            
        except Exception as e:
            logger.error(f"Error initializing memory metrics: {str(e)}")
            self._update_performance_metrics('errors_count', self.get_performance_metrics().get('errors_count', 0) + 1)

    def _get_memory_usage_mb(self):
        """
        Get current memory usage in MB with monitoring.
        """
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            memory_mb = mem_info.rss / (1024 * 1024)
            
            # Vérifier les seuils critiques
            if memory_mb > self.memory_config['memory_critical_threshold_mb']:
                logger.error(f"CRITICAL: Memory usage exceeds critical threshold: {memory_mb:.1f} MB")
                metrics = self.get_performance_metrics()
                warnings_count = metrics.get('warnings_count', 0)
                self._update_performance_metrics('warnings_count', warnings_count + 1)
            elif memory_mb > self.memory_config['memory_warning_threshold_mb']:
                logger.warning(f"Memory usage warning: {memory_mb:.1f} MB")
                metrics = self.get_performance_metrics()
                warnings_count = metrics.get('warnings_count', 0)
                self._update_performance_metrics('warnings_count', warnings_count + 1)
            
            return memory_mb
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            self._update_performance_metrics('errors_count', self.get_performance_metrics().get('errors_count', 0) + 1)
            return 0

    def _cleanup_memory(self):
        """
        Helper method to clean up memory with aggressive cleanup.
        """
        try:
            # Clear cached data
            if hasattr(self, 'current_chunk_data'):
                self.current_chunk_data = None
            
            # Clear scaler caches
            for scaler in self.scalers.values():
                if scaler is not None:
                    if hasattr(scaler, 'clear_cache'):
                        scaler.clear_cache()
            
            # Force garbage collection
            if self.memory_config['force_gc']:
                gc.collect()
                
            # Log memory usage
            current_memory = self._get_memory_usage_mb()
            if current_memory > self.memory_peak_mb:
                self.memory_peak_mb = current_memory
                self._update_performance_metrics('memory_peak_mb', self.memory_peak_mb)
            
            logger.info(f"Memory cleanup completed. Current usage: {current_memory:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            self._update_performance_metrics('errors_count', 1)

    def _update_performance_metrics(self, metric: str, value: Any) -> None:
        """
        Update performance metrics safely.
        
        Args:
            metric: The metric name to update
            value: The new value for the metric
        """
        if not hasattr(self, '_performance_metrics'):
            self._performance_metrics = {
                'gc_collections': 0,
                'memory_peak_mb': self.initial_memory_mb,
                'errors_count': 0,
                'warnings_count': 0
            }
        
        self._performance_metrics[metric] = value

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get the current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not hasattr(self, '_performance_metrics'):
            # Utiliser la mémoire initiale si elle est définie, sinon 0
            initial_memory = getattr(self, 'initial_memory_mb', 0)
            return {
                'gc_collections': 0,
                'memory_peak_mb': initial_memory,
                'errors_count': 0,
                'warnings_count': 0
            }
        return self._performance_metrics

    def _init_scalers(self):
        """
        Initialize scalers for each timeframe with advanced normalization.
        
        Each timeframe gets its own scaler with specific parameters:
        - 5m: MinMaxScaler with feature_range (-1, 1)
        - 1h: StandardScaler with mean=0, std=1
        - 4h: RobustScaler for outlier resistance
        
        Memory optimizations:
        - Use float32 for scaler parameters
        - Cache scaler parameters efficiently
        """
        # Nettoyer les scalers existants
        if self.scalers:
            for scaler in self.scalers.values():
                if scaler is not None:
                    del scaler
            self.scalers = {tf: None for tf in self.timeframes}
            gc.collect()
            
        # Initialiser les nouveaux scalers
        for tf in self.timeframes:
            if tf == "5m":
                self.scalers[tf] = MinMaxScaler(feature_range=(-1, 1), copy=False)
            elif tf == "1h":
                self.scalers[tf] = StandardScaler(copy=False)
            elif tf == "4h":
                self.scalers[tf] = RobustScaler(copy=False)
            else:
                self.scalers[tf] = StandardScaler(copy=False)
            
            # Optimiser la mémoire en utilisant float32
            if hasattr(self.scalers[tf], 'dtype'):
                self.scalers[tf].dtype = np.float32
            
        logger.info(f"Initialized scalers for timeframes: {list(self.scalers.keys())}")
        if not self.normalize:
            logger.info("Normalization disabled - no scalers initialized")
            return
            
        scaler_configs = {
            '5m': {'scaler_type': 'minmax', 'feature_range': (-1, 1)},
            '1h': {'scaler_type': 'standard'},
            '4h': {'scaler_type': 'robust'}
        }
        
        for tf in self.timeframes:
            config = scaler_configs.get(tf, {'scaler_type': 'standard'})
            
            if config['scaler_type'] == 'minmax':
                scaler = MinMaxScaler(feature_range=config.get('feature_range', (-1, 1)))
            elif config['scaler_type'] == 'standard':
                scaler = StandardScaler()
            elif config['scaler_type'] == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {config['scaler_type']}")
                
            self.scalers[tf] = scaler
            logger.info(f"Scaler initialized for timeframe {tf}: {config['scaler_type']} "
                        f"with params: {config}")

    def fit_scalers(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit scalers on the provided data with memory optimization.
        
        Args:
            data: Dictionary mapping timeframes to DataFrames
        """
        if not self.normalize:
            return
        
        logger.info("Fitting scalers on provided data...")
        
        # Vérifier la mémoire avant le fitting
        current_memory = self._get_memory_usage_mb()
        if current_memory > self.memory_config['memory_warning_threshold_mb']:
            logger.warning(f"Memory usage high before fitting: {current_memory:.1f} MB")
        
        try:
            for tf in self.timeframes:
                if tf not in self.scalers:
                    raise ValueError(f"Scaler not initialized for timeframe {tf}")
                if self.scalers[tf] is None:
                    if tf == "5m":
                        self.scalers[tf] = MinMaxScaler(feature_range=(-1, 1), copy=False)
                    elif tf == "1h":
                        self.scalers[tf] = StandardScaler(copy=False)
                    elif tf == "4h":
                        self.scalers[tf] = RobustScaler(copy=False)
                    else:
                        self.scalers[tf] = StandardScaler(copy=False)
                    logger.info(f"Initializing scaler for timeframe {tf}")
            
            for tf, df in data.items():
                if tf not in self.timeframes:
                    logger.warning(f"Skipping unknown timeframe {tf}")
                    continue
                
                columns = [col for col in self.features_config.get(tf, []) if col in df.columns]
                if not columns:
                    logger.warning(f"No matching feature columns found for timeframe {tf}")
                    continue
                
                # Utiliser float32 pour optimiser la mémoire
                timeframe_data = df[columns].values.astype(np.float32)
                
                if not np.isfinite(timeframe_data).all():
                    logger.warning(f"Non-finite values found in {tf} data. Replacing with zeros.")
                    timeframe_data = np.nan_to_num(timeframe_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                if self.scalers[tf] is None:
                    raise ValueError(f"Scaler not properly initialized for timeframe {tf}")
                
                if len(timeframe_data) < 2:
                    raise ValueError(f"Not enough data samples ({len(timeframe_data)}) to fit scaler for {tf}")
                
                # Fit le scaler avec optimisation de mémoire
                self.scalers[tf].fit(timeframe_data)
                logger.info(f"Fitted scaler for timeframe {tf} on {len(timeframe_data)} samples")
            
            # Sauvegarder les scalers si nécessaire
            if self.scaler_path:
                self.save_scalers()
            
            # Nettoyer la mémoire après le fitting
            if self.memory_config['aggressive_cleanup']:
                self._cleanup_memory()
                
        except Exception as e:
            logger.error(f"Error fitting scalers: {str(e)}")
            self._update_performance_metrics('errors_count', self.get_performance_metrics().get('errors_count', 0) + 1)
            raise
        
        # Mettre à jour les métriques de mémoire
        current_memory = self._get_memory_usage_mb()
        # Utiliser 0 comme valeur par défaut si memory_peak_mb n'est pas défini
        self.memory_peak_mb = max(getattr(self, 'memory_peak_mb', 0), current_memory)
        self._update_performance_metrics('memory_peak_mb', self.memory_peak_mb)

    def build_multi_channel_observation(self, 
                                        current_idx: int, 
                                        data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Build a multi-channel observation with all timeframes and memory optimization.
        
        Args:
            current_idx: Current index in the data
            data: Dictionary mapping timeframes to DataFrames
            
        Returns:
            3D numpy array of shape (n_timeframes, window_size, n_features)
            
        Raises:
            ValueError: If data is missing or insufficient
            KeyError: If required features are missing
            RuntimeError: If observation shape mismatch occurs
        """
        try:
            # Vérifier la mémoire avant le traitement
            current_memory = self._get_memory_usage_mb()
            if current_memory > self.memory_config['memory_warning_threshold_mb']:
                logger.warning(f"Memory usage high before building observation: {current_memory:.1f} MB")
            
            # Build observations for each timeframe
            observations = self.build_observation(current_idx, data)
            
            # Initialize output array with fixed shape
            output = np.zeros(self.observation_shape, dtype=np.float32)
            
            # Fill the output array with observations
            for i, (tf, obs) in enumerate(observations.items()):
                if obs is not None and len(obs) > 0:
                    # Take the most recent window_size observations
                    obs = obs[-self.window_size:]
                    
                    # Ensure correct number of features
                    if obs.shape[1] > self.max_features:
                        obs = obs[:, :self.max_features]
                    elif obs.shape[1] < self.max_features:
                        # Pad with zeros if needed
                        pad_width = ((0, 0), (0, self.max_features - obs.shape[1]))
                        obs = np.pad(obs, pad_width, mode='constant')
                    
                    # Handle window size
                    if obs.shape[0] < self.window_size:
                        # Pad with zeros at the beginning
                        pad_width = ((self.window_size - obs.shape[0], 0), (0, 0))
                        obs = np.pad(obs, pad_width, mode='constant')
                    elif obs.shape[0] > self.window_size:
                        # Take the most recent observations
                        obs = obs[-self.window_size:]
                    
                    # Store in output array
                    output[i] = obs
        
            # Mettre à jour les métriques de mémoire
            current_memory = self._get_memory_usage_mb()
            self.memory_peak_mb = max(getattr(self, 'memory_peak_mb', 0), current_memory)
            self._update_performance_metrics('memory_peak_mb', self.memory_peak_mb)
        
            return output
            
        except Exception as e:
            logger.error(f"Error building multi-channel observation: {str(e)}")
            raise
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Retourne la forme de l'observation.
        Forme fixe : (3, 100, 36) pour (timeframes, fenêtre, features)
        """
        return len(self.timeframes), self.window_size, self.max_features

    def calculate_expected_flat_dimension(self, portfolio_included: bool = False) -> int:
        """
        Calculate the expected flattened dimension of the observation state.
        
        Args:
            portfolio_included: Whether to include the portfolio state in the calculation.

        Returns:
            The total expected number of features in the flattened observation.
        """
        # The shape is (n_timeframes, window_size, max_features)
        n_timeframes, window_size, n_features = self.get_observation_shape()
        
        # For a flattened vector, the total dimension is channels * time * features
        total_dim = n_timeframes * window_size * n_features
        
        # This version may also include portfolio state
        if self.include_portfolio_state and portfolio_included:
            # This is a simplified placeholder. A real implementation would get this from a portfolio manager.
            # Based on build_portfolio_state, we have 7 base features + 5*2 position features = 17
            total_dim += 17 
            
        logger.info(f"Calculated expected flat dimension: {total_dim} (portfolio included: {portfolio_included})")
        return total_dim

    def validate_dimension(self, data: Dict[str, pd.DataFrame], portfolio_manager=None):
        """
        Validates the actual dimension of a built state against the expected dimension.

        Args:
            data: A sample data slice to build a test observation.
            portfolio_manager: An optional portfolio manager instance.

        Raises:
            ValueError: If the actual dimension does not match the expected dimension.
        """
        # We pass the portfolio_manager to the calculation method to decide if it should be included
        expected_dim = self.calculate_expected_flat_dimension(portfolio_manager is not None)
        
        # Build a sample state to get the actual dimension
        # We need a sample index, let's take the last one from the largest dataframe
        if not data:
            logger.warning("Cannot validate dimension without data.")
            return True
        
        max_len = max(len(df) for df in data.values())
        current_idx = max_len - 1
        
        test_observation_3d = self.build_multi_channel_observation(current_idx, data)
        
        if test_observation_3d is None:
            logger.warning("Could not build a sample observation for validation, skipping.")
            return True

        actual_dim = test_observation_3d.shape[0] * test_observation_3d.shape[1] * test_observation_3d.shape[2]

        if actual_dim != expected_dim:
            error_report = self._generate_error_report(actual_dim, expected_dim, portfolio_manager is not None)
            logger.error(f"Dimension mismatch detected:\n{error_report}")
            return False

        logger.info(f"Dimension validation passed: {actual_dim} == {expected_dim}")
        return True

    def _generate_error_report(self, actual_dim: int, expected_dim: int, portfolio_included: bool) -> Dict[str, Any]:
        """Generates a detailed report for a dimension mismatch error."""
        n_timeframes, window_size, n_features = self.get_observation_shape()
        market_contribution = n_timeframes * window_size * n_features
        
        portfolio_contribution = 0
        if self.include_portfolio_state and portfolio_included:
            portfolio_contribution = expected_dim - market_contribution

        discrepancy = actual_dim - expected_dim
        analysis = f"⚠️ System has a {-discrepancy} dimension discrepancy."

        return {
            "expected_dimension": expected_dim,
            "actual_dimension": actual_dim,
            "discrepancy": discrepancy,
            "discrepancy_analysis": analysis,
            "calculation_breakdown": {
                "observation_shape": self.get_observation_shape(),
                "market_data_contribution": market_contribution,
                "portfolio_contribution": portfolio_contribution,
                "window_size": self.window_size,
                "features_per_timeframe": self.nb_features_per_tf
            }
        }
    
    def build_portfolio_state(self, portfolio_manager: Any) -> np.ndarray:
        """
        Build portfolio state information to include in observations.
        
        Args:
            portfolio_manager: Portfolio manager instance
            
        Returns:
            Numpy array containing portfolio state information
        """
        if not self.include_portfolio_state or portfolio_manager is None:
            return np.zeros(17, dtype=np.float32)  # Return zero-padded portfolio state
        
        try:
            metrics = portfolio_manager.get_metrics()
            portfolio_state = [
                metrics.get('cash', 0.0),
                metrics.get('total_capital', 0.0),
                metrics.get('total_pnl_pct', 0.0),  # Using total_pnl_pct as returns
                metrics.get('sharpe_ratio', 0.0),
                metrics.get('drawdown', 0.0),
                len(metrics.get('positions', {})),
                ((metrics.get('total_capital', 0.0) - metrics.get('cash', 0.0)) /
                 metrics.get('total_capital', 0.0) if metrics.get('total_capital', 0.0) > 0 else 0.0)
            ]
            
            # Add individual position information (up to 5 largest positions)
            sorted_positions = sorted(metrics.get('positions', {}).items(), key=lambda x: abs(x[1].get('size', 0.0)), reverse=True)[:5]
            
            for i, (asset, position_obj) in enumerate(sorted_positions):
                portfolio_state.append(position_obj.get('size', 0.0))
                portfolio_state.append(hash(asset) % 1000)  # Simple asset encoding
            
            # Pad remaining position slots with zeros
            for i in range(len(sorted_positions), 5):
                portfolio_state.append(0.0)
                portfolio_state.append(0.0)
            
            return np.array(portfolio_state, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error building portfolio state: {e}")
            return np.zeros(17, dtype=np.float32)  # Return zero-padded portfolio state
    
    def validate_observation(self, observation: np.ndarray) -> bool:
        """
        Validate that an observation meets design specifications.
        
        Args:
            observation: Observation array to validate
            
        Returns:
            True if observation is valid, False otherwise
        """
        try:
            # Check shape according to design: (3, window_size, nb_features)
            if observation.shape[0] != len(self.timeframes):
                logger.error(f"Invalid observation shape: expected {len(self.timeframes)} timeframes, got {observation.shape[0]}")
                return False
            
            if observation.shape[1] != self.window_size:
                logger.error(f"Invalid observation shape: expected window size {self.window_size}, got {observation.shape[1]}")
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(observation).all():
                logger.error("Observation contains NaN or infinite values")
                return False
            
            # Check value ranges (normalized data should be roughly in [-3, 3] range)
            if self.normalize:
                if np.abs(observation).max() > 10:
                    logger.warning(f"Observation values seem unnormalized: max absolute value is {np.abs(observation).max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating observation: {e}")
            return False
    
    def get_feature_names(self, timeframe: str) -> List[str]:
        """
        Get feature names for a specific timeframe.
        
        Args:
            timeframe: Timeframe to get features for
            
        Returns:
            List of feature names
        """
        return self.features_config.get(timeframe, [])
    
    def reset_scalers(self) -> None:
        """Reset all scalers to unfitted state."""
        self._init_scalers()
        logger.info("All scalers have been reset")
    
    def get_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get normalization statistics from fitted scalers.
        
        Returns:
            Dictionary containing mean and scale for each timeframe
        """
        stats = {}
        
        for tf, scaler in self.scalers.items():
            if scaler is not None and hasattr(scaler, 'mean_'):
                stats[tf] = {
                    'mean': scaler.mean_,
                    'scale': scaler.scale_,
                    'var': getattr(scaler, 'var_', None)
                }
        
        return stats
    
    def calculate_market_volatility(self, data: Dict[str, pd.DataFrame], current_idx: int) -> float:
        """
        Calculate current market volatility for adaptive window sizing.
        
        Args:
            data: Dictionary mapping timeframes to DataFrames
            current_idx: Current index in the data
            
        Returns:
            Normalized volatility score (0.0 = low volatility, 1.0+ = high volatility)
        """
        try:
            volatilities = []
            
            for tf in self.timeframes:
                if tf not in data:
                    continue
                    
                df = data[tf]
                
                # Get recent price data for volatility calculation
                start_idx = max(0, current_idx - self.volatility_window + 1)
                price_window = df.iloc[start_idx:current_idx+1]
                
                # Use close price or high-low range for volatility
                if 'close' in price_window.columns:
                    prices = price_window['close']
                elif f'{tf}_close' in price_window.columns:
                    prices = price_window[f'{tf}_close']
                else:
                    continue
                
                if len(prices) < 2:
                    continue
                
                # Calculate returns and volatility
                returns = prices.pct_change().dropna()
                if len(returns) > 0:
                    vol = returns.std() * np.sqrt(len(returns))  # Annualized volatility
                    volatilities.append(vol * self.timeframe_weights[tf])
            
            if not volatilities:
                return 0.5  # Default medium volatility
            
            # Weighted average volatility
            weighted_volatility = np.mean(volatilities)
            
            # Update volatility history
            self.volatility_history.append(weighted_volatility)
            if len(self.volatility_history) > self.volatility_window:
                self.volatility_history.pop(0)
            
            # Normalize against historical volatility
            if len(self.volatility_history) > 1:
                hist_mean = np.mean(self.volatility_history)
                hist_std = np.std(self.volatility_history)
                
                if hist_std > 0:
                    normalized_vol = (weighted_volatility - hist_mean) / hist_std
                    # Convert to 0-1+ scale
                    normalized_vol = max(0, (normalized_vol + 2) / 4)  # Shift and scale
                else:
                    normalized_vol = 0.5
            else:
                normalized_vol = 0.5
            
            return min(2.0, normalized_vol)  # Cap at 2.0 for extreme volatility
            
        except Exception as e:
            logger.error(f"Error calculating market volatility: {e}")
            return 0.5  # Default medium volatility
    
    def adapt_window_size(self, volatility: float) -> int:
        """
        Adapt window size based on market volatility.
        
        Args:
            volatility: Normalized volatility score (0.0 to 2.0+)
            
        Returns:
            Adapted window size
        
        Raises:
            ValueError: If volatility is out of expected range
        """
        if not self.adaptive_window:
            return self.base_window_size
            
        if not (0.0 <= volatility <= 2.0):
            raise ValueError(f"Volatility score {volatility} out of expected range [0.0, 2.0]")
        
        # High volatility -> smaller window (more reactive)
        # Low volatility -> larger window (more stable)
        
        # Calculate window size based on volatility
        if volatility < 0.3:
            # Low volatility: use larger window for stability
            adapted_size = int(self.base_window_size * 1.5)
        elif volatility < 0.7:
            # Medium volatility: use base window size
            adapted_size = self.base_window_size
        else:
            # High volatility: use smaller window for reactivity
            adapted_size = int(self.base_window_size * 0.7)
            
        # Ensure window size stays within bounds
        adapted_size = max(self.min_window_size, min(adapted_size, self.max_window_size))
        
        # Log the adaptation
        logger.info(f"Adapting window size: base={self.base_window_size}, volatility={volatility:.2f}, adapted={adapted_size}")
        
        return adapted_size
    
    def update_adaptive_window(self, data: Dict[str, pd.DataFrame], current_idx: int) -> None:
        """
        Update the window size based on current market conditions.
        
        Args:
            data: Dictionary mapping timeframes to DataFrames
            current_idx: Current index in the data
            
        Raises:
            ValueError: If data is invalid or volatility calculation fails
        """
        if not self.adaptive_window:
            return
            
        try:
            # Calculate current market volatility
            volatility = self.calculate_market_volatility(data, current_idx)
            
            # Adapt window size
            new_window_size = self.adapt_window_size(volatility)
            
            # Update window size if it changed significantly (threshold of 10%)
            change_threshold = 0.10  # 10% change threshold
            if abs(new_window_size - self.window_size) > (self.window_size * change_threshold):
                old_size = self.window_size
                self.window_size = new_window_size
                logger.info(f"Adapted window size from {old_size} to {new_window_size} "
                           f"(volatility: {volatility:.3f}, change: {abs(new_window_size - old_size)} steps)")
                
        except Exception as e:
            logger.error(f"Error updating adaptive window: {e}")
            raise ValueError(f"Failed to update adaptive window: {e}")
    
    def apply_timeframe_weighting(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply intelligent weighting to different timeframes based on market conditions.
        
        Args:
            observations: Dictionary of observations by timeframe
            
        Returns:
            Weighted observations
        """
        weighted_observations = {}
        
        for tf, obs in observations.items():
            if tf in self.timeframe_weights:
                weight = self.timeframe_weights[tf]
                
                # Apply weight to the observation
                # For normalized data, we can scale the values
                weighted_obs = obs * weight
                
                # Ensure we don't lose important information by applying a minimum weight
                min_weight = 0.3
                if weight < min_weight:
                    weighted_obs = obs * min_weight + weighted_obs * (1 - min_weight)
                
                weighted_observations[tf] = weighted_obs
            else:
                weighted_observations[tf] = obs
        
        return weighted_observations
    
    def get_adaptive_stats(self) -> Dict[str, Union[int, float, List[float]]]:
        """
        Get statistics about the adaptive window system.
        
        Returns:
            Dictionary containing adaptive window statistics
        """
        return {
            'adaptive_enabled': self.adaptive_window,
            'base_window_size': self.base_window_size,
            'current_window_size': self.window_size,
            'min_window_size': self.min_window_size,
            'max_window_size': self.max_window_size,
            'volatility_history': self.volatility_history.copy(),
            'current_volatility': self.volatility_history[-1] if self.volatility_history else 0.0,
            'timeframe_weights': self.timeframe_weights.copy()
        }
    
    def build_observation(self, current_idx: int, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Build observations for each timeframe.
        
        Args:
            current_idx: Current index in the data
            data: Dictionary mapping timeframes to DataFrames
            
        Returns:
            Dictionary mapping timeframes to their observations
        """
        observations = {}
        
        for tf in self.timeframes:
            if tf not in data:
                raise KeyError(f"Missing data for timeframe {tf}")
                
            df = data[tf]
            features = self.features_config.get(tf, [])
            
            if not features:
                raise ValueError(f"No features configured for timeframe {tf}")
                
            if current_idx < self.window_size:
                # Retourner un tableau de zéros de la forme attendue si pas assez de données
                logger.warning(f"Current index {current_idx} is less than window_size {self.window_size} for timeframe {tf}. Returning zero-padded observation for {tf}.")
                observations[tf] = np.zeros((self.window_size, len(features)), dtype=np.float32)
                continue
                
            # Create a case-insensitive column mapping
            column_mapping = {col.upper(): col for col in df.columns}
            
            # Log available and requested features for debugging
            logger.debug(f"Available columns in {tf} data: {df.columns.tolist()}")
            logger.debug(f"Requested features for {tf}: {features}")
            
            # Map features to actual column names (case-insensitive)
            mapped_features = []
            for f in features:
                upper_f = f.upper()
                if upper_f in column_mapping:
                    mapped_features.append(column_mapping[upper_f])
                    logger.debug(f"Mapped feature: '{f}' -> '{column_mapping[upper_f]}'")
                else:
                    logger.warning(f"Feature '{f}' not found in DataFrame columns")
                    mapped_features.append(f)  # Will raise KeyError if not found
            
            # Get the window of data with mapped column names
            try:
                window_data = df[mapped_features].iloc[current_idx - self.window_size:current_idx]
            except KeyError as e:
                logger.error(f"Error accessing columns for {tf}: {e}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                logger.error(f"Requested columns: {mapped_features}")
                raise
            
            if window_data.empty:
                logger.warning(f"Window data is empty for timeframe {tf} at index {current_idx}. Returning zero-padded observation for {tf}.")
                observations[tf] = np.zeros((self.window_size, len(features)), dtype=np.float32)
                logger.debug(f"Empty window_data for {tf}. Shape: {window_data.shape}, Dtypes: {window_data.dtypes.to_dict()}")
                continue

            if window_data.isnull().values.any():
                logger.warning(f"NaN values found in {tf} data, replacing with zeros")
                window_data = window_data.fillna(0)
                
            # Convert to numpy array with float32 for memory optimization
            obs = window_data.values.astype(np.float32)
            logger.debug(f"Observation for {tf} before normalization. Shape: {obs.shape}, Dtype: {obs.dtype}, Content: {obs.flatten()[:5]}...")
            
            # Normalize if required
            if self.normalize and tf in self.scalers and self.scalers[tf] is not None:
                obs = self.scalers[tf].transform(obs)
                logger.debug(f"Observation for {tf} after normalization. Shape: {obs.shape}, Dtype: {obs.dtype}, Content: {obs.flatten()[:5]}...")
                
            observations[tf] = obs
            
        logger.debug(f"build_observation returning: type={type(observations)}, content={observations}")
        return observations

    def build_adaptive_observation(self, 
                                 current_idx: int, 
                                 data: Dict[str, pd.DataFrame],
                                 portfolio_manager: Any = None) -> np.ndarray:
        """
        Build observation with adaptive window sizing and timeframe weighting.
        
        Args:
            current_idx: Current index in the data
            data: Dictionary mapping timeframes to DataFrames
            
        Returns:
            3D numpy array with adaptive sizing and weighting applied
        """
        # Update adaptive window based on current market conditions
        self.update_adaptive_window(data, current_idx)
        
        # Build standard observations
        observations = self.build_observation(current_idx, data)
        
        logger.debug(f"build_adaptive_observation received: type={type(observations)}, content={observations}")
        if not observations or any(not isinstance(obs, np.ndarray) or obs.size == 0 for obs in observations.values()):
            logger.warning("Observations dictionary is empty or contains empty arrays. Returning zero-padded observation.")
            logger.debug(f"Observations before weighting: {observations}")
            return np.zeros(self.observation_shape, dtype=np.float32)
        
        # Apply timeframe weighting
        weighted_observations = self.apply_timeframe_weighting(observations)
        
        # Build multi-channel observation with current window size
        # Use self.max_features which is determined during initialization
        max_features = self.max_features
        n_timeframes = len(self.timeframes)
        
        # Initialize a 1D array for the flattened observation
        flattened_output = np.zeros(self.total_flattened_observation_size, dtype=np.float32)
        current_offset = 0
        
        # Fill the output array
        for i, tf in enumerate(self.timeframes):
            if tf in weighted_observations:
                obs = weighted_observations[tf]
                
                # Adjust observation to current window size
                if obs.shape[0] > self.window_size:
                    # Take the most recent data
                    obs = obs[-self.window_size:]
                elif obs.shape[0] < self.window_size:
                    # Pad with zeros at the beginning
                    padding = np.zeros((self.window_size - obs.shape[0], obs.shape[1]), dtype=np.float32)
                    obs = np.vstack([padding, obs])
                
                # Debug log the observation shape
                logger.debug(f"Observation for {tf}: shape={obs.shape}, features={obs.shape[1]}, max_features={max_features}")
                
                # Handle feature dimension padding
                if obs.shape[1] < max_features:
                    padding = max_features - obs.shape[1]
                    left_pad = padding // 2
                    right_pad = padding - left_pad
                    obs = np.pad(obs, ((0, 0), (left_pad, right_pad)), 
                               mode='constant', constant_values=0)
                    logger.debug(f"Padded {tf} observation from {obs.shape[1]-padding} to {obs.shape[1]} features")
                
                # Make sure the observation has the correct number of features
                if obs.shape[1] != max_features:
                    logger.warning(f"Observation for {tf} has {obs.shape[1]} features, expected {max_features}. Adjusting...")
                    obs = obs[:, :max_features]  # Truncate if too many features
                    
                # Flatten the current timeframe's observation and place it into the main flattened_output
                tf_flattened_obs = obs.flatten()
                expected_tf_size = self.window_size * max_features
                
                if tf_flattened_obs.size != expected_tf_size:
                    logger.warning(f"Timeframe {tf} flattened size mismatch: {tf_flattened_obs.size} vs expected {expected_tf_size}. Adjusting.")
                    if tf_flattened_obs.size < expected_tf_size:
                        tf_padding = np.zeros(expected_tf_size - tf_flattened_obs.size, dtype=tf_flattened_obs.dtype)
                        tf_flattened_obs = np.concatenate([tf_flattened_obs, tf_padding])
                    else:
                        tf_flattened_obs = tf_flattened_obs[:expected_tf_size]
                
                current_offset += expected_tf_size

        # Add portfolio state to the end of the flattened output
        if self.include_portfolio_state:
            portfolio_state_array = self.build_portfolio_state(portfolio_manager)
            
            # Ensure portfolio state array has the expected size (17 features)
            expected_portfolio_size = 17
            if portfolio_state_array.size != expected_portfolio_size:
                logger.warning(f"Portfolio state size mismatch. Expected {expected_portfolio_size}, got {portfolio_state_array.size}. Adjusting.")
                if portfolio_state_array.size < expected_portfolio_size:
                    portfolio_padding = np.zeros(expected_portfolio_size - portfolio_state_array.size, dtype=np.float32)
                    portfolio_state_array = np.concatenate([portfolio_state_array, portfolio_padding])
                else:
                    portfolio_state_array = portfolio_state_array[:expected_portfolio_size]

            # Append portfolio state to the flattened output
            flattened_output[current_offset : current_offset + expected_portfolio_size] = portfolio_state_array
            current_offset += expected_portfolio_size

        # Final check on the total size of the flattened output
        if flattened_output.size != self.total_flattened_observation_size:
            logger.error(f"Final flattened output size mismatch. Expected {self.total_flattened_observation_size}, got {flattened_output.size}. Adjusting.")
            if flattened_output.size < self.total_flattened_observation_size:
                final_padding = np.zeros(self.total_flattened_observation_size - flattened_output.size, dtype=np.float32)
                flattened_output = np.concatenate([flattened_output, final_padding])
            else:
                flattened_output = flattened_output[:self.total_flattened_observation_size]

        return flattened_output
