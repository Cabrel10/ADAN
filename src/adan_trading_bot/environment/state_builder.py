import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json
from ..common.utils import get_logger

logger = get_logger()

class StateBuilder:
    def __init__(self, config, assets, encoder=None, base_feature_names=None, cnn_input_window_size=None):
        self.config = config
        self.assets = assets
        self.encoder = encoder
        self.initial_capital = config.get('environment', {}).get('initial_capital', 10000.0)
        self.training_timeframe = config.get('data', {}).get('training_timeframe', '1h')
        self.base_feature_names = base_feature_names if base_feature_names is not None else \
                                  config.get('data', {}).get('base_market_features', ['open', 'high', 'low', 'close', 'volume'])
        self.cnn_input_window_size = cnn_input_window_size if cnn_input_window_size is not None else \
                                     config.get('data', {}).get('cnn_input_window_size', 20)

        # Load global scaler and its feature order
        self.global_scaler = None
        self.global_scaler_feature_order = None
        scaler_base_path = Path(config.get('paths', {}).get('base_project_dir_local', '.')) / \
                           config.get('paths', {}).get('scalers_encoders_dir', 'data/scalers_encoders')
        scaler_path = scaler_base_path / f'global_scaler_{self.training_timeframe}.joblib'
        feature_order_path = scaler_base_path / f'global_scaler_{self.training_timeframe}_feature_order.json'

        if scaler_path.exists():
            try: self.global_scaler = joblib.load(scaler_path)
            except Exception as e: logger.error(f"StateBuilder failed to load scaler {scaler_path}: {e}")
        else: logger.warning(f"StateBuilder: Global scaler not found at {scaler_path}.")

        if feature_order_path.exists():
            try:
                with open(feature_order_path, 'r') as f: self.global_scaler_feature_order = json.load(f)
                logger.info(f"StateBuilder loaded feature order ({len(self.global_scaler_feature_order)} features) from {feature_order_path}")
            except Exception as e: logger.error(f"StateBuilder failed to load feature order {feature_order_path}: {e}")
        else: logger.warning(f"StateBuilder: Feature order file not found at {feature_order_path}.")

        # MODIFIED: Construct canonical_fallback_feature_order with timeframe for indicators
        self.canonical_fallback_feature_order = []
        ohlcv_base = ['open', 'high', 'low', 'close', 'volume']
        for asset_item in self.assets: # Asset major
            for base_feature_item in self.base_feature_names: # Feature minor
                if base_feature_item in ohlcv_base:
                    self.canonical_fallback_feature_order.append(f"{base_feature_item}_{asset_item}")
                else: # It's an indicator
                    self.canonical_fallback_feature_order.append(f"{base_feature_item}_{self.training_timeframe}_{asset_item}")
        
        if not self.global_scaler_feature_order:
            logger.warning(f"StateBuilder: Using canonical fallback feature order ({len(self.canonical_fallback_feature_order)} features).")
        
        self.typical_position_sizes = {asset: self.initial_capital / 10.0 for asset in assets}
        logger.info(f"🏗️ StateBuilder: {len(assets)} assets, {len(self.base_feature_names)} base features, window={self.cnn_input_window_size}, tf={self.training_timeframe}")

    def build_observation(self, market_data_window, capital, positions, image_shape=None): # Removed apply_scaling
        # apply_scaling is now determined internally by presence of self.global_scaler
        image_features = self._get_market_features_as_image(market_data_window, image_shape)
        portfolio_features = self._get_portfolio_features(capital, positions)
        return {"image_features": image_features.astype(np.float32), "vector_features": portfolio_features.astype(np.float32)}

    def _get_market_features_as_image(self, market_data_window, image_shape=None):
        num_channels = 1
        window_size = self.cnn_input_window_size
        
        feature_order_to_use = self.global_scaler_feature_order if self.global_scaler_feature_order else self.canonical_fallback_feature_order
        if not feature_order_to_use:
            logger.error("StateBuilder: No feature order defined. Cannot build image tensor.")
            # Fallback to a shape based on base_feature_names * assets if absolutely necessary
            num_fallback_cols = len(self.base_feature_names) * len(self.assets) if self.base_feature_names and self.assets else 1
            return np.zeros((num_channels, window_size, num_fallback_cols), dtype=np.float32)

        num_image_columns = len(feature_order_to_use)
        image_tensor = np.zeros((window_size, num_image_columns), dtype=np.float32)
        actual_window_size = min(len(market_data_window), window_size)
        padding_offset = window_size - actual_window_size

        missing_features_log = []
        for idx, full_col_name in enumerate(feature_order_to_use):
            if full_col_name in market_data_window.columns:
                feature_values = market_data_window[full_col_name].values[-actual_window_size:]
                image_tensor[padding_offset:, idx] = np.asarray(feature_values, dtype=np.float32)
            else:
                image_tensor[padding_offset:, idx] = np.full(actual_window_size, np.nan, dtype=np.float32) # Fill with NaN first
                missing_features_log.append(full_col_name)
        
        if missing_features_log:
            logger.warning(f"StateBuilder: {len(missing_features_log)} features from order NOT FOUND in market_data_window (filled with NaN): {missing_features_log[:5]}")

        if np.isnan(image_tensor).any():
            image_tensor = np.nan_to_num(image_tensor, nan=0.0, posinf=1e6, neginf=-1e6) # Impute NaNs

        if self.global_scaler and self.global_scaler_feature_order : # Only scale if scaler AND its order are loaded
            if image_tensor.shape[1] == len(self.global_scaler_feature_order): # And shape matches
                try:
                    image_tensor = self.global_scaler.transform(image_tensor)
                except Exception as e:
                    logger.error(f"StateBuilder: Error applying global_scaler: {e}.")
            else:
                logger.error(f"StateBuilder: Shape mismatch for scaling. Image tensor {image_tensor.shape[1]} features, scaler expects {len(self.global_scaler_feature_order)}. Scaling skipped.")
        elif self.global_scaler and not self.global_scaler_feature_order:
             logger.warning("StateBuilder: Global scaler loaded, but its feature order is missing. Scaling skipped to avoid errors.")


        # Reshape: (num_channels, window_size, num_image_columns)
        # Use image_tensor.shape[1] for width, as it's based on feature_order_to_use
        final_image_width = image_tensor.shape[1]
        image_tensor_reshaped = image_tensor.reshape(num_channels, window_size, final_image_width)
        
        # Final check on shape if image_shape was provided
        if image_shape is not None and image_tensor_reshaped.shape != image_shape:
            logger.warning(f"StateBuilder: Final image tensor shape {image_tensor_reshaped.shape} differs from provided image_shape {image_shape}. Using actual tensor shape.")
            
        return image_tensor_reshaped

    def _get_portfolio_features(self, capital, positions):
        capped_capital = min(capital, 1e6)
        normalized_capital = min(capped_capital / self.initial_capital, 100.0)
        position_features = np.zeros(len(self.assets), dtype=np.float32)
        for i, asset in enumerate(self.assets):
            if asset in positions:
                try:
                    typical_size = self.typical_position_sizes.get(asset, self.initial_capital / 10.0)
                    position_qty = min(positions[asset]["qty"], 1e8)
                    normalized_position = position_qty / typical_size if typical_size != 0 else 0
                    position_features[i] = np.clip(normalized_position, -100.0, 100.0)
                except Exception as e: logger.warning(f"Error normalizing position for {asset}: {e}"); position_features[i] = 0.0
        
        portfolio_features = np.concatenate([[normalized_capital], position_features])
        if not np.all(np.isfinite(portfolio_features)):
            portfolio_features = np.nan_to_num(portfolio_features, nan=0.0, posinf=100.0, neginf=-100.0)
        return portfolio_features

    def get_observation_space_dim(self):
        num_channels = 1
        window_size = self.cnn_input_window_size
        
        feature_order = self.global_scaler_feature_order if self.global_scaler_feature_order else self.canonical_fallback_feature_order
        num_features_per_step = len(feature_order) if feature_order else \
                                (len(self.base_feature_names) * len(self.assets) if self.base_feature_names and self.assets else 1)
        
        image_shape = (num_channels, window_size, num_features_per_step)
        portfolio_dim = 1 + len(self.assets)
        logger.info(f"StateBuilder Observation space: Image={image_shape}, Vector={portfolio_dim}")
        return {"image_features": image_shape, "vector_features": portfolio_dim}
