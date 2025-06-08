"""
State builder for the ADAN trading environment.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json # Added
from ..common.utils import get_logger

logger = get_logger()

class StateBuilder:
    """
    Builds observation states for the RL agent based on market data and portfolio state.
    """
    
    def __init__(self, config, assets, encoder=None, base_feature_names=None, cnn_input_window_size=None): # Removed scaler
        """
        Initialize the state builder.
        
        Args:
            config: Configuration dictionary.
            assets: List of asset symbols.
            # scaler: Optional pre-fitted scaler for market features. (Removed)
            encoder: Optional pre-fitted encoder for dimensionality reduction.
            base_feature_names: List of base feature names to use for the CNN image.
            cnn_input_window_size: Size of the window for CNN input.
        """
        self.config = config
        self.assets = assets
        # self.scaler = scaler # Removed
        self.encoder = encoder
        self.initial_capital = config.get('environment', {}).get('initial_capital', 10000.0)

        self.global_scaler = None # Initialize
        timeframe = self.config.get('data', {}).get('training_timeframe', '1h') # Default or from config

        # Construct path to scaler
        # Use project_root from config if available, otherwise assume relative path for data
        project_root_str = self.config.get('paths', {}).get('base_project_dir_local', '.')
        scalers_encoders_path_str = self.config.get('paths', {}).get('scalers_encoders_dir', 'data/scalers_encoders')

        # Ensure scalers_encoders_path_str is treated as relative to project_root if not absolute
        scaler_base_path = Path(scalers_encoders_path_str)
        if not scaler_base_path.is_absolute():
            scaler_base_path = Path(project_root_str) / scalers_encoders_path_str

        scaler_path = scaler_base_path / f'global_scaler_{timeframe}.joblib'
        
        if scaler_path.exists():
            try:
                self.global_scaler = joblib.load(scaler_path)
                logger.info(f"StateBuilder loaded global scaler from {scaler_path}")
            except Exception as e:
                logger.error(f"StateBuilder failed to load global scaler from {scaler_path}: {e}. Scaling will be skipped.")
                self.global_scaler = None
        else:
            logger.warning(f"StateBuilder: Global scaler not found at {scaler_path}. Market features will not be scaled by StateBuilder.")

        self.global_scaler_feature_order = None
        if self.global_scaler: # Only try to load feature order if scaler was loaded
            feature_order_filename = f'global_scaler_{timeframe}_feature_order.json'
            # scaler_path is already defined and is a Path object
            feature_order_path = scaler_path.parent / feature_order_filename

            if feature_order_path.exists():
                try:
                    with open(feature_order_path, 'r') as f:
                        self.global_scaler_feature_order = json.load(f)
                    logger.info(f"StateBuilder loaded global scaler feature order ({len(self.global_scaler_feature_order)} features) from {feature_order_path}")
                except Exception as e:
                    logger.error(f"StateBuilder failed to load global scaler feature order from {feature_order_path}: {e}. Will rely on fallback order if scaling.")
                    self.global_scaler_feature_order = None # Ensure it's None on failure
            else:
                logger.warning(f"StateBuilder: Global scaler feature order file not found at {feature_order_path}. Relying on fallback order if scaling.")

        # Get the list of base market features to use (needed for fallback order construction)
        if base_feature_names is not None:
            self.base_feature_names = base_feature_names
            # logger.debug(f"📊 Features from param: {len(self.base_feature_names)}") # Already logged if base_feature_names is None path taken before
        else:
            self.base_feature_names = config.get('data', {}).get('base_market_features',
                                                          ['open', 'high', 'low', 'close', 'volume', 'macd'])
            # logger.debug(f"📊 Features from config: {len(self.base_feature_names)}")

        # If global_scaler_feature_order could not be loaded, create a canonical fallback order.
        # This order is crucial for determining the shape of the image tensor and for iterating
        # through features if the .json list is missing.
        # The actual application of the scaler should ideally ONLY happen if self.global_scaler_feature_order is present.
        self.canonical_fallback_feature_order = []
        for asset_item in self.assets: # Iterate assets first
            for base_feature_item in self.base_feature_names: # Then features
                # This constructs feature names like open_ADAUSDT, rsi_1h_ADAUSDT etc.
                # This is one possible way to order them if the JSON is missing.
                # Note: This order (AssetMajor then FeatureMajor) is DIFFERENT from what the global scaler likely expects
                # (which is often FeatureMajor then AssetMajor from DataFrame column order).
                # This fallback is primarily for defining the shape if JSON is missing, not for guaranteeing correct scaling.
                self.canonical_fallback_feature_order.append(f"{base_feature_item}_{asset_item}")

        if not self.global_scaler_feature_order:
            logger.warning(f"StateBuilder: Using a fallback canonical feature order of {len(self.canonical_fallback_feature_order)} features. If global_scaler is present and used, this fallback order might lead to incorrect scaling due to feature order mismatch.")
        else:
            logger.info(f"StateBuilder: Will use loaded global_scaler_feature_order for constructing market image if applicable.")
            if len(self.global_scaler_feature_order) != len(self.canonical_fallback_feature_order):
                logger.warning(f"Mismatch in length between loaded feature order ({len(self.global_scaler_feature_order)}) and fallback order ({len(self.canonical_fallback_feature_order)}). This could indicate issues if fallback is ever used with scaler.")

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
        # Pass apply_scaling parameter
        image_features = self._get_market_features_as_image(market_data_window, image_shape, apply_scaling=apply_scaling)
        
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
        logger.debug(f"🔍 Extract: {market_data_window.shape}, {len(self.base_feature_names)}×{len(self.assets)} features, ApplyScaling: {apply_scaling}")
        
        # Determine image shape
        if image_shape is None:
            num_channels = 1
            window_size = self.cnn_input_window_size
            if window_size is None:
                window_size = self.config.get('data', {}).get('cnn_input_window_size', 20)
                self.cnn_input_window_size = window_size
            
            # Determine the order of features for the image tensor
            feature_order_to_use = self.global_scaler_feature_order if self.global_scaler_feature_order else self.canonical_fallback_feature_order
            
            if not feature_order_to_use:
                logger.error("StateBuilder: No feature order defined (neither from JSON nor fallback). Cannot build image tensor.")
                # Return a zero tensor with a guess for shape or a predefined error shape
                num_fallback_features = len(self.base_feature_names) * len(self.assets) if self.base_feature_names and self.assets else 1
                return np.zeros((num_channels if image_shape else 1,
                                 window_size if image_shape else self.cnn_input_window_size,
                                 num_fallback_features), dtype=np.float32)

            num_image_columns = len(feature_order_to_use)

            # Update image_shape's width if it was passed in and mismatches, or log
            if image_shape is not None and image_shape[2] != num_image_columns:
                logger.warning(f"StateBuilder: image_shape parameter width {image_shape[2]} mismatches determined feature order width {num_image_columns}. Using {num_image_columns}.")
                num_features = num_image_columns # Correct num_features based on the order to be used
            elif image_shape is None:
                num_features = num_image_columns # num_features now reflects the actual columns we'll build
            # else image_shape[2] == num_image_columns, so num_features is already correct.

            # Initialize image tensor
            image_tensor = np.zeros((window_size, num_image_columns), dtype=np.float32)
            actual_window_size = min(len(market_data_window), window_size)
            padding_offset = window_size - actual_window_size
            
            missing_critical_features_count = 0
            for idx, full_col_name in enumerate(feature_order_to_use):
                if full_col_name in market_data_window.columns:
                    feature_values = market_data_window[full_col_name].values[-actual_window_size:]
                    image_tensor[padding_offset:, idx] = np.asarray(feature_values, dtype=np.float32)
                else:
                    image_tensor[padding_offset:, idx] = np.full(actual_window_size, np.nan, dtype=np.float32)
                    logger.warning(f"StateBuilder: Feature '{full_col_name}' from feature_order not found in market_data_window. Filled with NaN.")
                    missing_critical_features_count +=1
            
            if missing_critical_features_count > 0:
                logger.error(f"StateBuilder: {missing_critical_features_count} features from the defined order were NOT FOUND in market_data_window.")

        else: # market_data_window is not a DataFrame
            logger.error(f"❌ Unexpected type for market_data_window: {type(market_data_window)}")
            # Determine num_image_columns for zero tensor based on available feature order
            feature_order_to_use = self.global_scaler_feature_order if self.global_scaler_feature_order else self.canonical_fallback_feature_order
            num_image_columns = len(feature_order_to_use) if feature_order_to_use else (len(self.base_feature_names) * len(self.assets) if self.base_feature_names and self.assets else 1)
            image_tensor = np.zeros((window_size, num_image_columns), dtype=np.float32)

        # Impute NaNs (e.g., from missing columns or original data) BEFORE potentially scaling
        if np.isnan(image_tensor).any():
            logger.warning("StateBuilder: NaN values found in image_tensor before scaling (e.g. missing features). Imputing with 0.")
            image_tensor = np.nan_to_num(image_tensor, nan=0.0, posinf=1e6, neginf=-1e6)

        # Apply global scaler if available and requested
        if apply_scaling:
            if self.global_scaler is not None:
                if self.global_scaler_feature_order is None:
                    # This case means we loaded a scaler but not its feature order. Scaling here is highly risky.
                    logger.error("StateBuilder: Attempting to apply global_scaler, but its feature order JSON was not loaded. Scaling with potentially misaligned features based on canonical_fallback_feature_order. THIS IS RISKY.")
                # Check if the image_tensor width matches the scaler's expected features
                # The number of features the scaler was fit on is implicitly len(self.global_scaler_feature_order) if loaded,
                # or for the fallback, it's len(self.canonical_fallback_feature_order).
                # The image_tensor was built using one of these, so its width image_tensor.shape[1] should match.

                expected_scaler_features = 0
                if self.global_scaler_feature_order:
                    expected_scaler_features = len(self.global_scaler_feature_order)
                elif hasattr(self.global_scaler, 'n_features_in_'): # Scikit-learn scalers
                    expected_scaler_features = self.global_scaler.n_features_in_

                if expected_scaler_features > 0 and image_tensor.shape[1] != expected_scaler_features :
                     logger.error(f"StateBuilder: Shape mismatch for scaling. Image tensor has {image_tensor.shape[1]} features, but global_scaler was expecting {expected_scaler_features} (based on feature_order list or scaler attribute). Scaling skipped.")
                else:
                    try:
                        image_tensor = self.global_scaler.transform(image_tensor) # image_tensor is (window_size, num_features)
                        logger.debug("StateBuilder: Global scaler applied to market features.")
                    except Exception as e:
                        logger.error(f"StateBuilder: Error applying global_scaler: {e}. Features remain unscaled or partially scaled.")
            else: # self.global_scaler is None
                logger.warning("StateBuilder: 'apply_scaling' is True, but no global_scaler loaded. Features will not be scaled by StateBuilder.")
        else: # apply_scaling is False
            logger.info("StateBuilder: apply_scaling is False. Market features will not be scaled by StateBuilder.")

        # Reshape to include channel dimension (1, window_size, num_image_columns)
        # num_features here should be num_image_columns now
        if image_shape is None: # If original image_shape was None, use the determined num_image_columns
            final_image_shape_width = image_tensor.shape[1]
        else: # If image_shape was provided, its width might have been adjusted if feature_order_to_use changed it
            final_image_shape_width = num_features # num_features was updated if image_shape's width mismatched feature_order_to_use

        image_tensor = image_tensor.reshape(num_channels, window_size, final_image_shape_width)
        
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
        
        # Calculer le nombre de features par pas de temps based on the determined feature order
        if self.global_scaler_feature_order:
            num_features_per_step = len(self.global_scaler_feature_order)
            logger.info(f"Observation space width based on global_scaler_feature_order: {num_features_per_step}")
        elif hasattr(self, 'canonical_fallback_feature_order') and self.canonical_fallback_feature_order:
            num_features_per_step = len(self.canonical_fallback_feature_order)
            logger.info(f"Observation space width based on canonical_fallback_feature_order: {num_features_per_step}")
        else:
            # This case should ideally not be reached if __init__ always sets one of the orders.
            logger.warning("StateBuilder.get_observation_space_dim: Feature order not determined. Using default calculation (base_feature_names * assets).")
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
