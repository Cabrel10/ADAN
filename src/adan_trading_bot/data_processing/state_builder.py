"""
State builder for creating multi-timeframe observations for the RL agent.

This module provides the StateBuilder class which transforms raw market data
into a structured observation space suitable for reinforcement learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            timeframe: The timeframe identifier (e.g., '5m', '15m')
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
                 max_window_size: int = 30):  # 150% de la taille de fenêtre par défaut
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
        """
        # Configuration initiale
        self.features_config = features_config or self._get_extended_features_config()
        self.timeframes = list(self.features_config.keys())
        self.nb_features_per_tf = {tf: len(features) for tf, features in self.features_config.items()}
        
        # Configuration de la taille de fenêtre
        self.base_window_size = window_size
        self.window_size = window_size
        self.include_portfolio_state = include_portfolio_state
        self.normalize = normalize
        
        # Configuration adaptative
        self.adaptive_window = adaptive_window
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.volatility_history = []
        self.volatility_window = 20
        
        # Configuration des scalers
        self.scaler_path = scaler_path
        self.scalers = {}
        self.feature_indices = {}
        
        # Initialisation des scalers et calcul de la forme d'observation
        self._init_scalers()
        
        # Calcul de la forme d'observation
        # Structure 3D: (timeframes, window_size, features)
        # Ajout d'un canal pour le portfolio si activé
        observation_channels = len(self.timeframes)
        if self.include_portfolio_state:
            observation_channels += 1
            
        max_features = max(self.nb_features_per_tf.values())
        self.observation_shape = (observation_channels, self.base_window_size, max_features)
        
        logger.info(f"Observation shape configured as: {self.observation_shape}")
        
        logger.info(f"StateBuilder initialized with base_window_size={window_size}, "
                   f"adaptive_window={adaptive_window}, "
                   f"timeframes={self.timeframes}, "
                   f"features_per_timeframe={self.nb_features_per_tf}")

    def _init_scalers(self):
        """
        Initialize scalers for each timeframe with advanced normalization.
        
        Each timeframe gets its own scaler with specific parameters:
        - 5m: MinMaxScaler with feature_range (-1, 1)
        - 1h: StandardScaler with mean=0, std=1
        - 4h: RobustScaler for outlier resistance
        """
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
    
    def _get_extended_features_config(self) -> Dict[str, List[str]]:
        """
        Génère la configuration étendue des features incluant les 22+ indicateurs techniques.
        
        Returns:
            Configuration des features par timeframe
        
        Note:
            Cette configuration doit correspondre exactement à celle définie dans config.yaml
        """
        # Configuration des features par timeframe, conforme à config.yaml
        return {
            '5m': [
                'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME',
                'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
                'CCI_20_0.015', 'ROC_9', 'MFI_14',
                'EMA_5', 'EMA_20', 'SUPERTREND_14_2.0',
                'PSAR_0.02_0.2'
            ],
            '1h': [
                'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME',
                'RSI_14', 'MACD_12_26_9', 'MACD_HIST_12_26_9',
                'CCI_20_0.015', 'MFI_14',
                'EMA_50', 'EMA_100', 'SMA_200',
                'ICHIMOKU_9_26_52', 'PSAR_0.02_0.2'
            ],
            '4h': [
                'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME',
                'RSI_14', 'MACD_12_26_9', 'CCI_20_0.015',
                'MFI_14', 'EMA_50', 'SMA_200',
                'ICHIMOKU_9_26_52', 'SUPERTREND_14_3.0',
                'PSAR_0.02_0.2'
            ]
        }
        volume_indicators = [
            'OBV', 'VWAP', 'MFI', 'Volume_SMA', 'Volume_Ratio'
        ]
        
        # Combiner tous les indicateurs
        all_indicators = (trend_indicators + momentum_indicators + 
                         volatility_indicators + volume_indicators)
        
        # Configuration par timeframe
        config = {}
        for tf in self.timeframes:
            # Features de base avec suffixe timeframe
            tf_base_features = [f'{tf}_{feature}' for feature in base_features]
            
            # Indicateurs avec suffixe timeframe
            tf_indicators = [f'{indicator}_{tf}' for indicator in all_indicators]
            
            # Combiner et limiter si nécessaire
            all_features = tf_base_features + tf_indicators
            
            # Sélection dynamique des features si activée
            if self.feature_selection_enabled and len(all_features) > self.max_features_per_timeframe:
                # Prioriser les features les plus importantes
                priority_features = (tf_base_features + 
                                   [f'{ind}_{tf}' for ind in ['RSI', 'MACD', 'ATR', 'SMA_20', 'EMA_12']])
                
                # Ajouter d'autres features jusqu'à la limite
                remaining_slots = self.max_features_per_timeframe - len(priority_features)
                other_features = [f for f in tf_indicators if f not in priority_features]
                
                config[tf] = priority_features + other_features[:remaining_slots]
            else:
                config[tf] = all_features
        
        return config
    
    def auto_detect_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Détecte automatiquement les features disponibles dans les données.
        
        Args:
            data: Dictionnaire des DataFrames par timeframe
            
        Returns:
            Configuration des features détectées
        """
        detected_config = {}
        
        for tf in self.timeframes:
            if tf not in data:
                continue
                
            df = data[tf]
            available_columns = df.columns.tolist()
            
            # Filtrer les colonnes qui correspondent au timeframe
            tf_columns = [col for col in available_columns 
                         if col.startswith(f'{tf}_') or col.endswith(f'_{tf}')]
            
            # Ajouter les colonnes de base si elles existent
            base_columns = ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update']
            for base_col in base_columns:
                if base_col in available_columns:
                    tf_columns.append(base_col)
            
            # Limiter si nécessaire
            if self.feature_selection_enabled and len(tf_columns) > self.max_features_per_timeframe:
                tf_columns = tf_columns[:self.max_features_per_timeframe]
            
            detected_config[tf] = tf_columns
            logger.info(f"Detected {len(tf_columns)} features for {tf}: {tf_columns[:5]}...")
        
        return detected_config
    
    def update_features_config(self, new_config: Dict[str, List[str]]) -> None:
        """
        Met à jour la configuration des features.
        
        Args:
            new_config: Nouvelle configuration des features
        """
        self.features_config = new_config
        self.nb_features_per_tf = {tf: len(features) for tf, features in self.features_config.items()}
        
        # Recalculer la shape des observations
        max_features = max(self.nb_features_per_tf.values())
        self.observation_shape = (len(self.timeframes), self.window_size, max_features)
        
        # Réinitialiser les scalers
        self._init_scalers()
        
        logger.info(f"Updated features config. New observation shape: {self.observation_shape}")
    
    def build_observation(self, current_idx: int, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Construit une observation 3D à partir des données.
        
        Args:
            current_idx: Index actuel dans les données
            data: Dictionnaire des DataFrames par timeframe
            
        Returns:
            Observation 3D de forme (timeframes, window_size, n_features)
        
        Raises:
            ValueError: Si les données sont manquantes ou insuffisantes
            KeyError: Si les features configurées ne sont pas présentes dans les données
        """
        observations = {}
        
        try:
            # Vérifier la présence des timeframes requis
            missing_timeframes = set(self.timeframes) - set(data.keys())
            if missing_timeframes:
                raise ValueError(f"Missing required timeframes: {missing_timeframes}")
            
            # Vérifier la présence des features pour chaque timeframe
            for tf in self.timeframes:
                missing_features = set(self.features_config[tf]) - set(data[tf].columns)
                if missing_features:
                    raise KeyError(f"Missing features for timeframe {tf}: {missing_features}")
            
            # Construire l'observation pour chaque timeframe
            for tf in self.timeframes:
                df = data[tf]
                
                # Vérifier que l'index est valide
                if current_idx < self.window_size:
                    raise ValueError(f"Current index {current_idx} is less than window size {self.window_size}")
                
                # Extraire les valeurs pour la fenêtre
                start_idx = current_idx - self.window_size
                values = df.iloc[start_idx:current_idx][self.features_config[tf]].values
                
                # Vérifier la présence des données
                if len(values) < self.window_size:
                    padding = np.zeros((self.window_size - len(values), len(self.features_config[tf])))
                    values = np.vstack([padding, values])
                
                # Normaliser si activé
                if self.normalize and self.scalers.get(tf) is not None:
                    try:
                        values = self.scalers[tf].transform(values)
                    except Exception as e:
                        logger.error(f"Error normalizing data for {tf}: {str(e)}")
                        raise
                
                observations[tf] = values
            
            return observations
            
        except Exception as e:
            logger.error(f"Error building observation: {str(e)}")
            raise
    
    def get_feature_importance_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Analyse l'importance des features basée sur leur variance et corrélation.
        
        Args:
            data: Dictionnaire des DataFrames par timeframe
            
        Returns:
            Dictionnaire d'importance des features par timeframe
        """
        importance_analysis = {}
        
        for tf in self.timeframes:
            if tf not in data:
                continue
                
            df = data[tf]
            expected_features = self.features_config.get(tf, [])
            available_features = [col for col in expected_features if col in df.columns]
            
            if not available_features:
                continue
            
            feature_data = df[available_features].select_dtypes(include=[np.number])
            
            if feature_data.empty:
                continue
            
            # Calculer l'importance basée sur la variance (features avec plus de variance sont plus informatives)
            variances = feature_data.var()
            
            # Normaliser les variances
            max_var = variances.max()
            if max_var > 0:
                normalized_variances = variances / max_var
            else:
                normalized_variances = variances
            
            # Calculer la corrélation moyenne avec les autres features (diversité)
            corr_matrix = feature_data.corr().abs()
            avg_correlations = corr_matrix.mean()
            
            # Score d'importance combiné (variance élevée, corrélation modérée)
            importance_scores = {}
            for feature in available_features:
                if feature in normalized_variances.index and feature in avg_correlations.index:
                    variance_score = normalized_variances[feature]
                    correlation_penalty = avg_correlations[feature] * 0.5  # Pénaliser forte corrélation
                    importance_scores[feature] = variance_score - correlation_penalty
                else:
                    importance_scores[feature] = 0.0
            
            importance_analysis[tf] = importance_scores
        
        return importance_analysis
    
    def _init_scalers(self) -> None:
        """Initialize scalers for each timeframe with advanced normalization."""
        for tf in self.timeframes:
            if self.normalize:
                # Use RobustScaler for better outlier handling
                from sklearn.preprocessing import RobustScaler
                self.scalers[tf] = RobustScaler(quantile_range=(25.0, 75.0))
            else:
                self.scalers[tf] = None
        
        # Cross-timeframe normalization parameters
        self.cross_tf_scaler = None
        self.outlier_detection_enabled = True
        self.outlier_threshold = 3.0  # Z-score threshold
        self.outlier_stats = {}
    
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
            expected_features = self.features_config.get(tf, [])
            columns = [col for col in expected_features if col in df.columns]
            if not columns:
                logger.warning(f"No matching feature columns found for timeframe {tf}")
                continue
                
            # Store the feature columns for this timeframe
            self.feature_indices[tf] = columns
            
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
        Build a multi-timeframe observation with robust validation.
        
        Args:
            current_idx: Current index in the data
            data: Dictionary mapping timeframes to DataFrames
            
        Returns:
            Dictionary mapping timeframes to observation arrays
            
        Raises:
            ValueError: If data is missing or insufficient
            KeyError: If required features are missing
        """
        try:
            observations = {}
            
            for tf, df in data.items():
                if tf not in self.timeframes:
                    continue
                
                # Validate data availability
                if df is None or df.empty:
                    raise ValueError(f"No data available for timeframe {tf}")
                    
                # Validate current index
                if current_idx >= len(df):
                    raise ValueError(f"Current index {current_idx} exceeds data length {len(df)} for {tf}")
                    
                # Get the window of data
                start_idx = max(0, current_idx - self.window_size + 1)
                df_window = df.iloc[start_idx:current_idx+1].copy()
                
                # Validate feature availability
                expected_features = self.features_config.get(tf, [])
                missing_features = [f for f in expected_features if f not in df_window.columns]
                if missing_features:
                    raise KeyError(f"Missing features for {tf}: {missing_features}")
                    
                # Get the feature values
                values = df_window[expected_features].values
                
                # Validate data shape
                if len(values) < self.window_size:
                    padding = np.zeros((self.window_size - len(values), len(expected_features)))
                    values = np.vstack([padding, values])
                    logger.warning(f"Insufficient history for {tf}, padding with zeros")
                
                # Apply normalization if enabled
                if self.normalize and self.scalers.get(tf) is not None:
                    try:
                        values = self.scalers[tf].transform(values)
                    except Exception as e:
                        logger.error(f"Error normalizing data for {tf}: {e}")
                        raise
                
                # Validate final shape
                expected_shape = (self.window_size, len(expected_features))
                if values.shape != expected_shape:
                    raise ValueError(f"Shape mismatch for {tf}: expected {expected_shape}, got {values.shape}")
                    
                # Add to observations
                observations[tf] = values
                
            # Validate that we have observations for all configured timeframes
            missing_tfs = set(self.timeframes) - set(observations.keys())
            if missing_tfs:
                raise ValueError(f"Missing observations for timeframes: {missing_tfs}")
                
            return observations
            
        except Exception as e:
            logger.error(f"Error building observation: {str(e)}")
            raise
    
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
            
        Raises:
            ValueError: If data is missing or insufficient
            KeyError: If required features are missing
            RuntimeError: If observation shape mismatch occurs
        """
        try:
            # Build observations for each timeframe
            observations = self.build_observation(current_idx, data)
            if not observations:
                raise ValueError("No observations built")
                
            # Validate observation shapes
            expected_shape = (self.window_size, len(self.features_config[tf]))
            for tf, obs in observations.items():
                if obs.shape != expected_shape:
                    raise ValueError(f"Shape mismatch for {tf}: expected {expected_shape}, got {obs.shape}")
                    
            # Find the maximum number of features across all timeframes
            max_features = max(obs.shape[1] for obs in observations.values())
            
            # Initialize the output array
            n_timeframes = len(self.timeframes)
            output = np.zeros((n_timeframes, self.window_size, max_features))
            
            # Fill the output array
            for i, tf in enumerate(self.timeframes):
                if tf not in observations:
                    raise KeyError(f"Missing observation for timeframe {tf}")
                    
                obs = observations[tf]
                
                # Center the features if they have fewer columns than max_features
                padding = max_features - obs.shape[1]
                if padding > 0:
                    left_pad = padding // 2
                    right_pad = padding - left_pad
                    obs = np.pad(obs, ((0, 0), (left_pad, right_pad)), 
                               mode='constant', constant_values=0)
                
                # Validate final placement
                if output[i, :, :].shape != obs.shape:
                    raise RuntimeError(f"Shape mismatch after padding for {tf}: "
                                     f"expected {output[i, :, :].shape}, got {obs.shape}")
                    
                output[i, :, :] = obs
            
            # Validate final output shape
            if output.shape != self.observation_shape:
                raise RuntimeError(f"Final observation shape mismatch: "
                                 f"expected {self.observation_shape}, got {output.shape}")
            
            return output
            
        except Exception as e:
            logger.error(f"Error building multi-channel observation: {str(e)}")
            raise
    
    def get_observation_shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of observations according to design specifications.
        
        Returns:
            Tuple representing (n_timeframes, window_size, n_features)
        """
        return self.observation_shape

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
    
    def build_portfolio_state(self, portfolio_manager) -> Dict[str, float]:
        """
        Build portfolio state information to include in observations.
        
        Args:
            portfolio_manager: Portfolio manager instance
            
        Returns:
            Dictionary containing portfolio state information
        """
        if not self.include_portfolio_state:
            return {}
        
        try:
            portfolio_state = {
                'cash': portfolio_manager.cash,
                'total_value': portfolio_manager.total_value,
                'returns': portfolio_manager.returns,
                'sharpe_ratio': getattr(portfolio_manager, 'sharpe_ratio', 0.0),
                'max_drawdown': getattr(portfolio_manager, 'max_drawdown', 0.0),
                'num_positions': len(portfolio_manager.positions),
                'position_value_ratio': (
                    (portfolio_manager.total_value - portfolio_manager.cash) / 
                    portfolio_manager.total_value if portfolio_manager.total_value > 0 else 0.0
                )
            }
            
            # Add individual position information (up to 5 largest positions)
            positions = getattr(portfolio_manager, 'positions', {})
            sorted_positions = sorted(positions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            for i, (asset, size) in enumerate(sorted_positions):
                portfolio_state[f'position_{i}_size'] = size
                portfolio_state[f'position_{i}_asset'] = hash(asset) % 1000  # Simple asset encoding
            
            # Pad remaining position slots with zeros
            for i in range(len(sorted_positions), 5):
                portfolio_state[f'position_{i}_size'] = 0.0
                portfolio_state[f'position_{i}_asset'] = 0.0
            
            return portfolio_state
            
        except Exception as e:
            logger.error(f"Error building portfolio state: {e}")
            return {}
    
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
    
    def build_adaptive_observation(self, 
                                 current_idx: int, 
                                 data: Dict[str, pd.DataFrame]) -> np.ndarray:
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
        
        if not observations:
            return None
        
        # Apply timeframe weighting
        weighted_observations = self.apply_timeframe_weighting(observations)
        
        # Build multi-channel observation with current window size
        max_features = max(obs.shape[1] for obs in weighted_observations.values())
        n_timeframes = len(self.timeframes)
        
        # Use current adaptive window size
        output = np.zeros((n_timeframes, self.window_size, max_features))
        
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
                    padding = np.zeros((self.window_size - obs.shape[0], obs.shape[1]))
                    obs = np.vstack([padding, obs])
                
                # Handle feature dimension padding
                if obs.shape[1] < max_features:
                    padding = max_features - obs.shape[1]
                    left_pad = padding // 2
                    right_pad = padding - left_pad
                    obs = np.pad(obs, ((0, 0), (left_pad, right_pad)), 
                               mode='constant', constant_values=0)
                
                output[i, :, :] = obs
        
        return output
