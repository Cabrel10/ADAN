"""
State builder for the ADAN trading environment.
"""
import numpy as np
import pandas as pd
from ..common.utils import get_logger

logger = get_logger()

class StateBuilder:
    """
    Builds observation states for the RL agent based on market data and portfolio state.
    """
    
    def __init__(self, config, assets, scaler=None, encoder=None, base_feature_names=None, cnn_input_window_size=None):
        """
        Initialize the state builder.
        
        Args:
            config: Configuration dictionary.
            assets: List of asset symbols.
            scaler: Optional pre-fitted scaler for market features.
            encoder: Optional pre-fitted encoder for dimensionality reduction.
            base_feature_names: List of base feature names to use for the CNN image.
            cnn_input_window_size: Size of the window for CNN input.
        """
        self.config = config
        self.assets = assets
        self.scaler = scaler
        self.encoder = encoder
        self.initial_capital = config.get('environment', {}).get('initial_capital', 10000.0)
        
        # Log encoder status
        encoder_status = "🔧 Enabled" if self.encoder is not None else "❌ Disabled"
        logger.info(f"🧠 Auto-encoder: {encoder_status}")
        
        # Get the training timeframe from config
        self.training_timeframe = config.get('data', {}).get('training_timeframe', '1h')
        
        # Get the list of base market features to use
        if base_feature_names is not None:
            self.base_feature_names = base_feature_names
            logger.debug(f"📊 Features from param: {len(self.base_feature_names)}")
        else:
            self.base_feature_names = config.get('data', {}).get('base_market_features', 
                                                          ['open', 'high', 'low', 'close', 'volume', 'macd'])
            logger.debug(f"📊 Features from config: {len(self.base_feature_names)}")
        
        # Get the list of numeric columns to use as market features (legacy support)
        self.numeric_cols = config.get('data', {}).get('numeric_cols', None)
        
        # Get CNN input window size
        if cnn_input_window_size is not None:
            self.cnn_input_window_size = cnn_input_window_size
        else:
            self.cnn_input_window_size = config.get('data', {}).get('cnn_input_window_size', 20)
        
        # Get training timeframe
        self.training_timeframe = config.get('data', {}).get('training_timeframe', '1h')
        
        # Set up typical position sizes for normalization
        self.typical_position_sizes = {asset: self.initial_capital / 10.0 for asset in assets}
        
        logger.info(f"🏗️ StateBuilder: {len(assets)} assets, {len(self.base_feature_names)} features, window={self.cnn_input_window_size}, tf={self.training_timeframe}")
    
    def build_observation(self, market_data_window, capital, positions, image_shape=None):
        """
        Build the observation dictionary for the current state.
        
        Args:
            market_data_window: Window of market data (DataFrame or dict of DataFrames).
            capital: Current capital.
            positions: Dictionary of current positions {asset_id: {"qty": quantity, "price": price}}.
            image_shape: Shape of the image features (channels, height, width).
            
        Returns:
            dict: Observation dictionary with 'image_features' and 'vector_features'.
        """
        # Extract market features as image
        image_features = self._get_market_features_as_image(market_data_window, image_shape)
        
        # Calculate portfolio features
        portfolio_features = self._get_portfolio_features(capital, positions)
        
        # Create observation dictionary
        observation = {
            "image_features": image_features.astype(np.float32),
            "vector_features": portfolio_features.astype(np.float32)
        }
        
        return observation
    
    def _get_market_features_as_image(self, market_data_window, image_shape=None):
        """
        Extract market features from the market data window and format them as an image tensor.
        
        Args:
            market_data_window: DataFrame with market features.
            image_shape: Tuple specifying (channels, height, width) for the image.
            
        Returns:
            numpy.ndarray: Image tensor with market features.
        """
        logger.debug(f"🔍 Extract: {market_data_window.shape}, {len(self.base_feature_names)}×{len(self.assets)} features")
        
        # Determine image shape
        if image_shape is None:
            num_channels = 1
            window_size = self.cnn_input_window_size
            if window_size is None:
                window_size = self.config.get('data', {}).get('cnn_input_window_size', 20)
                self.cnn_input_window_size = window_size
            
            num_features = len(self.base_feature_names) * len(self.assets)
            image_shape = (num_channels, window_size, num_features)
        else:
            num_channels, window_size, num_features = image_shape
        
        logger.debug(f"🖼️ Shape: {image_shape}, timesteps: {len(market_data_window)}")
        
        # Check if market_data_window is a DataFrame
        if isinstance(market_data_window, pd.DataFrame):
            # Initialize image tensor
            image_tensor = np.zeros((window_size, num_features), dtype=np.float32)
            
            # Determine actual window size (may be smaller than requested at the start of an episode)
            actual_window_size = min(len(market_data_window), window_size)
            padding_offset = window_size - actual_window_size
            
            # Track found and missing features for debugging
            found_features = []
            missing_features = []
            
            # Extract features for each asset and each base feature
            feature_idx = 0
            for asset in self.assets:
                for base_feature in self.base_feature_names:
                    if feature_idx >= num_features:
                        break
                    
                    # Try multiple column formats
                    column_found = False
                    column_to_find = None
                    
                    # Get timeframe from config
                    timeframe = self.config.get('data', {}).get('training_timeframe', '1m')
                    
                    # Pattern 1: {base_feature}_{asset}
                    pattern1 = f"{base_feature}_{asset}"
                    # Pattern 2: {base_feature}_{timeframe}_{asset}
                    pattern2 = f"{base_feature}_{timeframe}_{asset}"
                    
                    if pattern1 in market_data_window.columns:
                        column_to_find = pattern1
                        column_found = True
                    elif pattern2 in market_data_window.columns:
                        column_to_find = pattern2
                        column_found = True
                    
                    if column_found:
                        # Extract values from the last actual_window_size timesteps
                        feature_values = market_data_window[column_to_find].values[-actual_window_size:]
                        # Place values in the image with padding at the beginning
                        image_tensor[padding_offset:, feature_idx] = np.asarray(feature_values, dtype=np.float32)
                        found_features.append(column_to_find)
                    else:
                        # Feature not found - erreur critique
                        if len(missing_features) == 0:  # Log available columns only once
                            logger.error(f"❌ Feature '{pattern1}' or '{pattern2}' not found!")
                            logger.error(f"📋 Available columns: {market_data_window.columns.tolist()[:20]}")
                        
                        # Fill with NaN - pas de fallback
                        image_tensor[padding_offset:, feature_idx] = np.full(actual_window_size, np.nan, dtype=np.float32)
                        missing_features.append(f"{pattern1}/{pattern2}")
                    
                    feature_idx += 1
                
                if feature_idx >= num_features:
                    break
            
            # Log results
            if missing_features:
                logger.error(f"❌ Missing: {len(missing_features)}/{num_features} features - {missing_features[:3]}...")
            else:
                logger.debug(f"✅ All {len(found_features)} features found")
        else:
            logger.error(f"❌ Unexpected type for market_data_window: {type(market_data_window)}")
            image_tensor = np.zeros((window_size, num_features), dtype=np.float32)
        
        # Apply scaler if available
        if self.scaler is not None:
            try:
                # Reshape for scaling each time step
                original_shape = image_tensor.shape
                reshaped = image_tensor.reshape(-1, num_features)
                scaled = self.scaler.transform(reshaped)
                image_tensor = scaled.reshape(original_shape)
            except Exception as e:
                logger.error(f"❌ Scaler error: {e}")
                # Continue with unscaled data
        
        # Reshape to include channel dimension if needed
        if num_channels == 1:
            image_tensor = image_tensor.reshape(1, window_size, num_features)
        
        return image_tensor
        
    def _get_market_features(self, data_row):
        """
        Extract market features from the current data row (legacy support).
        
        Args:
            data_row: Current row of market data.
            
        Returns:
            numpy.ndarray: Market features.
        """
        # For merged data, we need to extract features for each asset
        all_features = []
        
        # Define base feature names to extract for each asset
        base_features = self.base_feature_names
        
        # Extract features for each asset
        for asset in self.assets:
            asset_features = []
            for base_feature in base_features:
                feature_name = f"{base_feature}_{asset}"
                if feature_name in data_row:
                    asset_features.append(data_row[feature_name])
                elif base_feature in data_row and 'pair' in data_row and data_row['pair'] == asset:
                    # Legacy format support
                    asset_features.append(data_row[base_feature])
                else:
                    # If feature not found, use 0 as placeholder
                    asset_features.append(0.0)
                    logger.debug(f"Feature {feature_name} not found for asset {asset}")
            
            all_features.extend(asset_features)
        
        # Convert to numpy array
        market_features = np.array(all_features, dtype=np.float32)
        
        # Apply scaler if available
        if self.scaler is not None:
            try:
                # Reshape for single sample
                market_features = market_features.reshape(1, -1)
                market_features = self.scaler.transform(market_features)
                market_features = market_features.flatten()
            except Exception as e:
                logger.error(f"Error applying scaler: {e}")
                # If transformation fails, use raw features
                market_features = np.array(all_features, dtype=np.float32)
        
        # Apply encoder if available
        if self.encoder is not None:
            try:
                # Reshape for single sample
                market_features = market_features.reshape(1, -1)
                market_features = self.encoder.predict(market_features)
                market_features = market_features.flatten()
            except Exception as e:
                logger.error(f"Error applying encoder: {e}")
                # If transformation fails, use features after scaler
        
        return market_features
    
    def _get_portfolio_features(self, capital, positions):
        """
        Calculate portfolio features.
        
        Args:
            capital: Current capital.
            positions: Dictionary of current positions.
            
        Returns:
            numpy.ndarray: Portfolio features.
        """
        # Plafonner le capital pour éviter les valeurs extrêmes
        MAX_CAPITAL = 1e6  # 1 million USD maximum
        capped_capital = min(capital, MAX_CAPITAL)
        
        # Normaliser le capital plafonné
        normalized_capital = capped_capital / self.initial_capital
        
        # Limiter la valeur normalisée pour éviter les valeurs extrêmes
        normalized_capital = min(normalized_capital, 100.0)
        
        # Initialize position features
        position_features = np.zeros(len(self.assets), dtype=np.float32)
        
        # Fill in position quantities
        for i, asset in enumerate(self.assets):
            if asset in positions:
                try:
                    # Normalize by typical position size
                    typical_size = self.typical_position_sizes.get(asset, self.initial_capital / 10.0)
                    
                    # Limiter la taille des positions pour éviter les overflows
                    MAX_POSITION_SIZE = 1e8  # 100 millions de jetons maximum
                    position_qty = min(positions[asset]["qty"], MAX_POSITION_SIZE)
                    
                    # Calculer la valeur normalisée
                    normalized_position = position_qty / typical_size
                    
                    # Limiter la valeur normalisée pour éviter les valeurs extrêmes
                    position_features[i] = np.clip(normalized_position, -100.0, 100.0)
                except (OverflowError, ZeroDivisionError, ValueError) as e:
                    logger.warning(f"Erreur lors de la normalisation de la position pour {asset}: {e}")
                    position_features[i] = 0.0
        
        # Combine capital and position features
        portfolio_features = np.concatenate([[normalized_capital], position_features])
        
        # Vérifier qu'il n'y a pas de valeurs NaN ou Inf
        if not np.all(np.isfinite(portfolio_features)):
            logger.warning(f"Valeurs non finies détectées dans portfolio_features: {portfolio_features}")
            # Remplacer les valeurs non finies par zéro
            portfolio_features = np.nan_to_num(portfolio_features, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return portfolio_features
    
    def get_observation_space_dim(self):
        """
        Get the dimension of the observation space.
        
        Returns:
            dict: Dictionary with dimensions for 'image_features' and 'vector_features'.
        """
        # Market features dimension (for image)
        num_channels = 1  # Par défaut, un seul canal
        window_size = self.cnn_input_window_size
        
        # Vérifier et journaliser la liste des features de base pour débogage
        logger.info(f"Base feature names ({len(self.base_feature_names)}): {self.base_feature_names}")
        logger.debug(f"Assets ({len(self.assets)}): {self.assets}")
        
        # Calculer le nombre de features par pas de temps
        num_features_per_step = len(self.base_feature_names) * len(self.assets)
        
        # Définir la forme du tenseur d'image
        image_shape = (num_channels, window_size, num_features_per_step)
        
        # Portfolio features: 1 (normalized capital) + len(assets) (normalized positions)
        portfolio_dim = 1 + len(self.assets)
        
        # Journaliser les dimensions pour débogage
        logger.info(f"Observation space dimensions:")
        logger.info(f"  - Image features: {image_shape} (channels, window_size, features)")
        logger.info(f"  - Vector features: {portfolio_dim}")
        
        return {
            "image_features": image_shape,
            "vector_features": portfolio_dim
        }
        
    def get_legacy_observation_space_dim(self):
        """
        Get the dimension of the legacy observation space (flat vector).
        
        Returns:
            int: Dimension of the legacy observation space.
        """
        # Market features dimension
        if self.encoder is not None:
            # If using an encoder, get its output dimension
            market_dim = self.encoder.output_shape[1]
        else:
            # Use base feature names multiplied by number of assets
            market_dim = len(self.base_feature_names) * len(self.assets)
        
        # Portfolio features: 1 (normalized capital) + len(assets) (normalized positions)
        portfolio_dim = 1 + len(self.assets)
        
        return market_dim + portfolio_dim
