#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environnement de trading multi-actifs avec chargement par morceaux.

Ce module implémente un environnement de trading pour plusieurs actifs
avec chargement efficace des données par lots.
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from omegaconf import OmegaConf, DictConfig

from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.data_processing.observation_validator import ObservationValidator
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
from adan_trading_bot.environment.order_manager import OrderManager
from adan_trading_bot.environment.reward_calculator import RewardCalculator

# Import TimeframeConfig depuis le bon module
try:
    from adan_trading_bot.data_processing.state_builder import TimeframeConfig
except ImportError:
    # Fallback si l'import échoue
    @dataclass
    class TimeframeConfig:
        """Configuration pour un timeframe spécifique."""
        timeframe: str
        features: List[str]
        window_size: int = 100

# Configuration du logger
logger = logging.getLogger(__name__)


class MultiAssetChunkedEnv(gym.Env):
    """
    Environnement de trading multi-actifs avec chargement par morceaux (chunks).
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    # Constantes pour les actions
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        config: Dict[str, Any],
        worker_config: Dict[str, Any],
        data_loader_instance: Optional[Any] = None,
        shared_buffer: Optional[Any] = None,
        worker_id: int = 0,
    ) -> None:
        """Initialise l'environnement de trading multi-actifs.

        Args:
            config: Configuration principale de l'application (déjà résolue).
            worker_config: Configuration spécifique au worker (déjà résolue).
            data_loader_instance: Instance de ChunkedDataLoader (optionnel).
            shared_buffer: Instance du SharedExperienceBuffer (optionnel).
        """
        super().__init__()

        self.config = config
        self.worker_config = worker_config
        self.data_loader_instance = data_loader_instance
        self.shared_buffer = shared_buffer

        # Initialize self.assets from worker_config
        self.assets = worker_config.get("assets", [])
        if not self.assets:
            raise ValueError("No assets specified in worker_config.")

        # Ajout d'un cache d'observations avec une taille maximale
        self._observation_cache = {}  # Cache for observations
        self._max_cache_size = 1000  # Nombre maximum d'observations en cache
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_access = {}  # Pour suivre la dernière utilisation

        # Initialisation des compteurs et états
        self.current_chunk = 0
        self.current_chunk_idx = 0
        self.done = False  # Ajout de l'état done
        
        # Initialisation des composants critiques
        self._initialized = False
        try:
            self._initialize_components()
            self._initialized = True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {str(e)}")
            raise

    def _initialize_components(self) -> None:
        """Initialize all environment components in the correct order."""
        # 1. Initialize data loader FIRST to know the data structure
        if self.data_loader_instance is not None:
            self.data_loader = self.data_loader_instance
        else:
            self._init_data_loader(self.assets)

        # 2. Dynamically create TimeframeConfig from the loaded data
        timeframe_configs = []
        if self.data_loader.features_by_timeframe:
            for tf_name, features in self.data_loader.features_by_timeframe.items():
                timeframe_configs.append(
                    TimeframeConfig(timeframe=tf_name, features=features, window_size=100)
                )
        else:
            raise ValueError("No feature configuration found in data loader.")

        # 3. Initialize portfolio manager
        portfolio_config = self.config.copy()
        env_config = self.config.get("environment", {})
        portfolio_config["trading_rules"] = self.config.get("trading_rules", {})
        portfolio_config["capital_tiers"] = self.config.get("capital_tiers", [])
        portfolio_config["initial_capital"] = env_config.get("initial_balance", 10000.0)
        
        # Map asset names to full names (e.g., BTC -> BTCUSDT)
        asset_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
        }
        mapped_assets = [asset_mapping.get(asset, asset) for asset in self.assets]
        
        # Initialize portfolio with mapped asset names
        self.portfolio = PortfolioManager(env_config=portfolio_config, assets=mapped_assets)
        
        # Update self.assets to use mapped names
        self.assets = mapped_assets

        # Convert list of TimeframeConfig objects to dictionary
        timeframe_configs_dict = {
            tf_config.timeframe: tf_config for tf_config in timeframe_configs
        }

        # 4. Initialize StateBuilder with the dynamic config
        features_config = {
            tf: config.features
            for tf, config in timeframe_configs_dict.items()
        }

        # Initialize with default target size first
        window_size = self.config.get("state", {}).get("max_window_size", 50)
        self.state_builder = StateBuilder(
            features_config=features_config,
            window_size=window_size,
            include_portfolio_state=True,
            normalize=True
        )

        # 5. Setup action and observation spaces (requires state_builder)
        self._setup_spaces()

        # 5. Initialize other components using worker_config where available
        trading_rules = self.config.get("trading_rules", {})
        penalties = self.config.get("environment", {}).get("penalties", {})
        self.order_manager = OrderManager(trading_rules=trading_rules, penalties=penalties)

        # Use reward_config from worker_config, fallback to main config's environment section
        reward_cfg = self.worker_config.get(
            "reward_config", self.config.get("environment", {}).get("reward_config", {})
        )
        # Create a proper env_config dictionary with the reward_shaping section
        env_config = {"reward_shaping": reward_cfg}
        self.reward_calculator = RewardCalculator(env_config=env_config)
        self.observation_validator = ObservationValidator()

        # Use dbe_config from worker_config, fallback to main config
        dbe_config = self.worker_config.get("dbe_config", self.config.get("dbe", {}))
        self.dbe = DynamicBehaviorEngine(
            config=dbe_config,  # Pass the specific DBE config
            finance_manager=getattr(self.portfolio, "finance_manager", None),
        )

    def _init_data_loader(self, assets: List[str]) -> Any:
        """Initialize the chunked data loader using worker-specific config.

        Returns:
            Initialized ChunkedDataLoader instance

        Raises:
            ValueError: If configuration is invalid or no assets are available
        """
        if not self.worker_config:
            raise ValueError("worker_config must be provided to initialize the data loader.")

        # Ensure paths are resolved
        if not hasattr(self, 'config') or not self.config:
            raise ValueError("Configuration not properly initialized")
        # Mapping for asset names to file system names (e.g., BTC -> BTCUSDT)
        asset_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
        }
        # Create a mapped assets list for the data loader
        mapped_assets = [
            asset_mapping.get(asset, asset) for asset in assets
        ]

        if not mapped_assets:
            raise ValueError("No assets specified in worker or environment config")

        # Nouveau code plus permissif
        global_data_timeframes = self.config.get("data", {}).get("timeframes", [])
        worker_timeframes = self.worker_config.get("timeframes", [])

        if worker_timeframes:
            # Le worker a spécifié quelques timeframes : on les prend tous
            self.timeframes = worker_timeframes
        else:
            # Sinon, on retombe sur la liste complète de la config globale
            self.timeframes = global_data_timeframes

        if not self.timeframes:
            raise ValueError(
                f"No timeframes defined: global={global_data_timeframes}, worker={worker_timeframes}"
            )

        # Create a copy of the config with resolved paths
        loader_config = {
            **self.config,
            "data": {
                **self.config.get("data", {}),
                "features_per_timeframe": self.config.get("data", {}).get("features_per_timeframe", {}),
                "assets": mapped_assets # Use mapped_assets here
            }
        }
        # Ensure we have features configured for each timeframe
        features_config = loader_config["data"]["features_per_timeframe"]
        for tf in self.timeframes:
            if tf not in features_config:
                logger.warning(f"No features configured for timeframe {tf}, using default features")
                features_config[tf] = ["open", "high", "low", "close", "volume"]

        # Initialize the data loader
        self.data_loader = ChunkedDataLoader(
            config=loader_config,
            worker_config={
                **self.worker_config,
                "assets": mapped_assets, # Use mapped_assets here
                "timeframes": self.timeframes
            }
        )

        

        # Get total chunks from the data loader
        self.total_chunks = self.data_loader.total_chunks
        logger.debug(f"MultiAssetChunkedEnv timeframes: {self.timeframes}")

        logger.info(
            f"Initialized data loader with {len(self.assets)} assets: {', '.join(self.assets)}"
        )
        logger.debug(f"Available timeframes: {', '.join(self.timeframes)}")

        return self.data_loader

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces.

        Raises:
            ValueError: If the observation space cannot be properly configured
        """
        # Action space: Continuous actions in [-1, 1] for each asset
        # -1 = max sell, 0 = hold, 1 = max buy
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.assets),),  # One action per asset
            dtype=np.float32,
        )

        try:
            # Get the 3D observation shape from StateBuilder
            n_timeframes, window_size, n_features = self.state_builder.get_observation_shape()
            # Store the shape for later use
            self.observation_shape = (n_timeframes, window_size, n_features)
            # Use the total flattened size calculated by StateBuilder
            flattened_size = self.state_builder.total_flattened_observation_size

            logger.info(f"Observation space shape: {self.observation_shape} (flattened: {flattened_size})")

            # Create a flat Box space that matches the actual flattened observation size
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(flattened_size,),  # Flattened shape for compatibility
                dtype=np.float32,
            )
            logger.info(f"Observation space configured with flattened shape: {flattened_size}")

        except Exception as e:
            logger.error(f"Failed to set up observation space: {str(e)}")
            raise

    def set_shared_buffer(self, shared_buffer):
        """
        Définit le buffer d'expérience partagé à utiliser par cet environnement.

        Args:
            shared_buffer: Instance de SharedExperienceBuffer à utiliser pour stocker
                         les expériences de cet environnement
        """
        self.shared_buffer = shared_buffer
        logger.info(f"Buffer d'expérience partagé défini pour l'environnement {self.worker_id}")

    def reset(self, *, seed=None, options=None):
        """Reset the environment to start a new episode.
        
        Args:
            seed: Optional seed for the random number generator
            options: Additional options for reset
            
        Returns:
            tuple: (observation, info) containing the initial observation and info
        """
        # Save cache information if needed
        cache_backup = {}
        if hasattr(self, '_observation_cache'):
            # Keep the last 10 observations
            cache_backup = dict(list(self._observation_cache.items())[-10:])

        # Initialize data structures
        self.current_data = {}
        self.current_step = 0
        self.step_in_chunk = 0
        self.current_chunk_idx = 0
        self.current_chunk = 0  # Initialisation de current_chunk
        self.episode = 0
        self.total_steps = 0
        # max_steps est déjà défini dans __init__, on l'utilise directement
        self.precomputed_windows = {}  # Dictionnaire pour stocker les fenêtres pré-calculées
        self._last_observation = None
        self._last_info = None
        self._last_reward = 0.0
        self._last_done = False
        self._last_truncated = False
        self._last_action = None
        # Suppression de la réinitialisation de _last_info car elle est déjà faite ci-dessus
        start_time = time.time()

        # Reset portfolio
        self.portfolio.reset()

        # Load the first chunk of data
        self.current_chunk_idx = 0
        self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)
        
        # Fit scalers on the first chunk of data
        if hasattr(self.state_builder, 'fit_scalers'):
            if callable(self.state_builder.fit_scalers):
                logger.info("Fitting scalers on initial data...")
                try:
                    # Prepare data in the format expected by fit_scalers
                    scaler_data = {}
                    for asset, timeframes in self.current_data.items():
                        for tf, df in timeframes.items():
                            if tf not in scaler_data:
                                scaler_data[tf] = []
                            scaler_data[tf].append(df)
                    
                    # Combine data for each timeframe
                    combined_data = {}
                    for tf, dfs in scaler_data.items():
                        # Only concatenate if there are DataFrames to combine
                        if dfs:
                            combined_data[tf] = pd.concat(dfs, axis=0)
                    
                    # Fit scalers on the combined data
                    if combined_data:
                        self.state_builder.fit_scalers(combined_data)
                        logger.info("Successfully fitted scalers on initial data")
                    else:
                        logger.warning("No data available to fit scalers")
                except Exception as e:
                    logger.error("Error fitting scalers: %s", str(e))
                    raise

        # Determine the total number of chunks from the loaded data
        # This is now initialized in _init_data_loader
        # if self.current_data:
        #     first_asset = next(iter(self.current_data.keys()))
        #     if first_asset and self.current_data[first_asset]:
        #         # Assuming chunk size is the length of the first dataframe
        #         self.total_chunks = 1  # Simplified for now
        #     else:
        #         self.total_chunks = 0
        # else:
        #     self.total_chunks = 0

        # Improved warm-up logic
        state_config = self.config.get("state", {})
        env_config = self.config.get("environment", {})
        self.window_size = state_config.get("window_size", 50)
        self.warmup_steps = env_config.get("warmup_steps", self.window_size)

        if self.warmup_steps < self.window_size:
            logger.warning(
                f"warmup_steps ({self.warmup_steps}) is less than "
                f"window_size ({self.window_size}). Setting warmup_steps to "
                f"{self.window_size}"
            )
            self.warmup_steps = self.window_size

        first_asset = next(iter(self.current_data.keys()))
        first_timeframe = next(iter(self.current_data[first_asset].keys()))
        data_length = len(self.current_data[first_asset][first_timeframe])

        if data_length < self.warmup_steps:
            raise ValueError(
                f"Le premier chunk ({data_length} steps) est plus petit "
                f"que la période de warm-up requise "
                f"({self.warmup_steps} steps)."
            )

        self.step_in_chunk = 0

        for _ in range(self.warmup_steps - 1):
            self.step_in_chunk += 1
            self.current_step += 1
            if self.step_in_chunk >= data_length:
                self.current_chunk_idx += 1
                if self.current_chunk_idx >= self.total_chunks:
                    raise ValueError(
                        "Reached end of data during warm-up period. "
                        f"Current chunk: {self.current_chunk_idx}, "
                        f"Total chunks: {self.data_loader.total_chunks}"
                    )
                self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)
                self.step_in_chunk = 0
                first_asset = next(iter(self.current_data.keys()))
                first_timeframe = next(iter(self.current_data[first_asset].keys()))
                data_length = len(self.current_data[first_asset][first_timeframe])

        observation = self._get_observation()
        info = self._get_info()
        
        

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        self._last_action = action

        if self.done:
            # Reset automatiquement l'environnement si l'épisode est terminé
            logger.info("Episode ended. Automatically resetting environment.")
            initial_obs, info = self.reset()
            return initial_obs, 0.0, False, False, info

        current_observation = self._get_observation()

        try:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

            if action.shape != (len(self.assets),):
                raise ValueError(
                    f"Action shape {action.shape} does not match "
                    f"expected shape (n_assets={len(self.assets)},)"
                )

            self._update_dbe_state()
            dbe_modulation = self.dbe.compute_dynamic_modulation()
            
            trade_start_time = time.time()
            self._execute_trades(action, dbe_modulation)
            trade_end_time = time.time()
            logger.debug(f"_execute_trades took {trade_end_time - trade_start_time:.4f} seconds")

            self.step_in_chunk += 1
            self.current_step += 1

            first_asset = next(iter(self.current_data.keys()))
            data_length = len(self.current_data[first_asset])

            if self.step_in_chunk >= data_length:
                self.current_chunk_idx += 1
                self.current_chunk += 1  # Incrémenter le compteur de chunks
                if self.current_chunk_idx >= self.total_chunks:
                    self.done = True
                else:
                    self.current_data = self.data_loader.load_chunk(
                        self.current_chunk_idx
                    )
                    self.step_in_chunk = 0
                    if hasattr(self.dbe, "_reset_for_new_chunk"):
                        self.dbe._reset_for_new_chunk()

            next_observation = self._get_observation()
            
            

            reward = self._calculate_reward(action)
            terminated = self.done
            truncated = False

            max_steps = getattr(self, "_max_episode_steps", float("inf"))
            if self.current_step >= max_steps:
                truncated = True
                self.done = True

            info = self._get_info()

            if hasattr(self, "_last_reward_components"):
                info.update({"reward_components": self._last_reward_components})

            if self.shared_buffer is not None:
                experience = {
                    'state': current_observation,
                    'action': action,
                    'reward': float(reward),
                    'next_state': next_observation,
                    'done': terminated or truncated,
                    'info': info,
                    'timestamp': self._get_safe_timestamp() or str(self.current_step),
                    'worker_id': self.worker_id
                }
                self.shared_buffer.add(experience)

            return next_observation, float(reward), terminated, truncated, info

        except Exception as e:
            logger.error(f"Error in step(): {str(e)}", exc_info=True)
            self.done = True
            observation = self._get_observation()
            info = self._get_info()
            info["error"] = str(e)
            return observation, 0.0, True, False, info

    def _update_dbe_state(self) -> None:
        """Update the DBE state with current market conditions."""
        try:
            current_prices = self._get_current_prices()
            portfolio_metrics = self.portfolio.get_metrics()

            live_metrics = {
                "step": self.current_step,
                "current_prices": current_prices,
                "portfolio_value": portfolio_metrics.get("total_capital", 0.0),
                "cash": portfolio_metrics.get("cash", 0.0),
                "positions": portfolio_metrics.get("positions", {}),
                "returns": portfolio_metrics.get("returns", 0.0),
                "max_drawdown": portfolio_metrics.get("max_drawdown", 0.0),
            }

            if hasattr(self, 'current_data') and self.current_data:
                first_asset = next(iter(self.current_data.keys()))
                if first_asset in self.current_data and self.current_data[first_asset]:
                    first_tf = next(iter(self.current_data[first_asset].keys()))
                    df = self.current_data[first_asset][first_tf]

                    if not df.empty and self.current_step < len(df):
                        current_row = df.iloc[self.current_step]
                        live_metrics.update(
                            {
                                "rsi": current_row.get("rsi", 50.0),
                                "adx": current_row.get("adx", 20.0),
                                "atr": current_row.get("atr", 0.0),
                                "atr_pct": current_row.get("atr_pct", 0.0),
                                "ema_ratio": current_row.get("ema_ratio", 1.0),
                            }
                        )
            if hasattr(self, 'dbe'):
                self.dbe.update_state(live_metrics)

        except Exception as e:
            logger.warning(f"Failed to update DBE state: {e}")

    def _execute_trades(
        self, action: np.ndarray, dbe_modulation: Dict[str, Any] = None
    ) -> None:
        """Execute trades based on the continuous action vector."""
        if len(action) != len(self.assets):
            raise ValueError(
                f"Action dimension {len(action)} does not match "
                f"number of assets {len(self.assets)}"
            )

        current_prices = self._get_current_prices()
        portfolio_metrics = self.portfolio.get_metrics()
        cash = portfolio_metrics.get("cash", 0.0)

        if dbe_modulation is None:
            dbe_modulation = {}

        for _i, (asset, action_value) in enumerate(
            zip(self.assets, action, strict=True)
        ):
            action_value = np.clip(float(action_value), -1.0, 1.0)

            if action_value > 0.1:
                discrete_action = 1
            elif action_value < -0.1:
                discrete_action = 2
            else:
                discrete_action = 0

            if discrete_action == 1:  # BUY
                if not self.portfolio.positions[asset].is_open:
                    self.order_manager.open_position(self.portfolio, asset, current_prices[asset], confidence=action_value)
                else:
                    logger.debug(f"Attempted to BUY {asset} but position is already open. Holding.")
            elif discrete_action == 2:  # SELL
                if self.portfolio.positions[asset].is_open:
                    self.order_manager.close_position(self.portfolio, asset, current_prices[asset])
                else:
                    logger.debug(f"Attempted to SELL {asset} but no position is open. Holding.")

        self.portfolio.update_market_price(current_prices)
        self.portfolio.rebalance(current_prices)

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all assets."""
        start_time = time.time()
        prices = {}
        for _asset, timeframe_data in self.current_data.items():
            if not timeframe_data:
                continue
            tf = next(iter(timeframe_data.keys()))
            df = timeframe_data[tf]
            if not df.empty and self.current_step < len(df):
                if "close" in df.columns:
                    prices[_asset] = df.iloc[self.current_step]["close"]
                elif not df.empty:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        current_row = df.iloc[self.current_step]
                        prices[_asset] = current_row[numeric_cols[0]]
        end_time = time.time()
        logger.debug(f"_get_current_prices took {end_time - start_time:.4f} seconds")
        return prices

    def _get_current_timestamp(self) -> pd.Timestamp:
        """Get the current timestamp."""
        for _asset, timeframe_data in self.current_data.items():
            if not timeframe_data:
                continue
            tf = next(iter(timeframe_data.keys()))
            df = timeframe_data[tf]
            if not df.empty and self.current_step < len(df):
                return df.index[self.current_step]
        raise RuntimeError("No timestamp data available")

    def _get_safe_timestamp(self) -> Optional[str]:
        """Get the current timestamp safely."""
        try:
            return self._get_current_timestamp().isoformat()
        except Exception:
            return None

    def _manage_cache(self, key: str, value: np.ndarray = None) -> Optional[np.ndarray]:
        """Gère le cache d'observations avec une politique LRU."""
        if key in self._observation_cache:
            self._cache_access[key] = time.time()
            self._cache_hits += 1
            return self._observation_cache[key]

        self._cache_misses += 1

        if len(self._observation_cache) >= self._max_cache_size:
            sorted_keys = sorted(
                self._cache_access.keys(),
                key=lambda k: self._cache_access[k]
            )
            num_to_remove = max(1, int(self._max_cache_size * 0.1))
            for k in sorted_keys[:num_to_remove]:
                self._observation_cache.pop(k, None)
                self._cache_access.pop(k, None)

        if value is not None:
            self._observation_cache[key] = value
            self._cache_access[key] = time.time()

        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation du cache."""
        total = self._cache_hits + self._cache_misses
        hit_ratio = self._cache_hits / total if total > 0 else 0.0

        return {
            'cache_enabled': True,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._observation_cache),
            'max_size': self._max_cache_size,
            'hit_ratio': hit_ratio
        }

    def _process_assets(self, feature_config: Dict[str, List[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Process asset data for the current step.
        
        Args:
            feature_config: Dictionary mapping timeframes to lists of feature names
            
        Returns:
            Dictionary mapping timeframes to dictionaries of {asset: DataFrame}
        """
        processed_data = {}
        
        for timeframe in self.timeframes:
            if timeframe not in feature_config:
                logger.warning(f"No feature configuration for timeframe {timeframe}")
                continue
                
            features = feature_config[timeframe]
            if not features:
                logger.warning(f"No features specified for timeframe {timeframe}")
                continue
                
            # Initialize dict to store processed DataFrames for this timeframe
            asset_data_dict = {}
            has_valid_data = False
            
            for asset in self.assets:
                # Get data for this asset and timeframe
                asset_data = self.current_data.get(asset, {}).get(timeframe)
                if asset_data is None or asset_data.empty:
                    logger.debug(f"No data for {asset} {timeframe}")
                    continue
                
                # Log all available columns for debugging
                logger.debug(f"Available columns in {asset} {timeframe} data: {asset_data.columns.tolist()}")
                logger.debug(f"Requested features for {timeframe}: {features}")
                
                # Create a mapping of uppercase column names to actual column names
                column_mapping = {col.upper(): col for col in asset_data.columns}
                
                # Find available features in the asset data (case-insensitive)
                available_features = []
                missing_features = []
                for f in features:
                    upper_f = f.upper()
                    if upper_f in column_mapping:
                        available_features.append(column_mapping[upper_f])
                        logger.debug(f"Found feature: '{f}' -> '{column_mapping[upper_f]}'")
                    else:
                        missing_features.append(f)
                        logger.debug(f"Missing feature: '{f}' (not in DataFrame columns)")
                
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} features for {asset} {timeframe}: {missing_features}")
                    logger.debug(f"Available columns in {asset} {timeframe}: {asset_data.columns.tolist()}")
                    logger.debug(f"Available features: {available_features}")
                
                if not available_features:
                    logger.warning(f"None of the requested features found for {asset} {timeframe}")
                    continue
                    
                try:
                    # Select only the requested features using their original case
                    asset_df = asset_data[available_features].copy()
                    
                    # Ensure column names are in uppercase for consistency
                    asset_df.columns = [col.upper() for col in asset_df.columns]
                    
                    # Store the DataFrame in the asset dictionary
                    asset_data_dict[asset] = asset_df
                    has_valid_data = True
                    
                except Exception as e:
                    logger.error(f"Error processing {asset} {timeframe}: {str(e)}")
                    logger.debug(f"Available columns: {asset_data.columns.tolist()}")
                    logger.debug(f"Available features: {available_features}")
            
            # Only add to processed_data if we have valid data for at least one asset
            if has_valid_data:
                processed_data[timeframe] = asset_data_dict
            else:
                logger.warning(f"No valid data for any asset in timeframe {timeframe}")
        
        return processed_data

    def _create_empty_dataframe(self, timeframe: str, window_size: int = None) -> pd.DataFrame:
        """Create an empty DataFrame with required features for a given timeframe.
        
        Args:
            timeframe: The timeframe for which to create the empty DataFrame
            window_size: Number of rows to include in the empty DataFrame
            
        Returns:
            DataFrame with required columns and zero values
        """
        try:
            # Get required features for this timeframe
            features = self.state_builder.get_feature_names(timeframe)
            if not features:
                logger.warning(f"No features defined for timeframe {timeframe}")
                features = ['close']  # Fallback to basic column
                
            # Use window_size if provided, otherwise default to 1
            rows = window_size if window_size is not None else 1
            
            # Create empty DataFrame with required features
            empty_data = np.zeros((rows, len(features)))
            df = pd.DataFrame(empty_data, columns=features)
            
            # Add timestamp column if not present
            if 'timestamp' not in df.columns and 'timestamp' in self.features[timeframe]:
                df['timestamp'] = pd.Timestamp.now()
                
            logger.debug(f"Created empty DataFrame for {timeframe} with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating empty DataFrame for {timeframe}: {str(e)}")
            # Fallback to minimal DataFrame
            return pd.DataFrame(columns=['timestamp', 'close'])

    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the environment."""
        start_time = time.time()
        
        # Vérifier si l'observation est dans le cache
        cache_key = f"{self.current_step}_{self.step_in_chunk}"
        cached_obs = self._manage_cache(cache_key)
        if cached_obs is not None:
            end_time = time.time()
            logger.debug(f"_get_observation (cached) took {end_time - start_time:.4f} seconds")
            return cached_obs

        try:
            # Log precomputed windows for debugging
            logger.debug("=== Precomputed Windows ===")
            for tf, assets in self.precomputed_windows.items():
                logger.debug(f"Timeframe {tf} assets: {list(assets.keys())}")
                for asset, window in assets.items():
                    if window is None:
                        logger.warning(f"  {asset}: None")
                    else:
                        logger.debug(f"  {asset}: shape={window.shape if hasattr(window, 'shape') else 'N/A'}")

            # Get feature configuration for all timeframes
            feature_config = {}
            for tf in self.timeframes:
                try:
                    features = self.state_builder.get_feature_names(tf)
                    if features:
                        feature_config[tf] = features
                        logger.debug(f"Configured features for {tf}: {features}")
                    else:
                        logger.warning(f"No features found for timeframe {tf}")
                except Exception as e:
                    logger.warning(f"Error getting features for timeframe {tf}: {str(e)}")
            
            if not feature_config:
                raise ValueError("No valid feature configurations found for any timeframe")

            # Process asset data for available timeframes
            processed_data = self._process_assets(feature_config)
            
            # Log processed data structure
            logger.debug("=== Processed Data Structure ===")
            for tf, assets in processed_data.items():
                logger.debug(f"Timeframe {tf} has data for assets: {list(assets.keys())}")
                for asset, df in assets.items():
                    logger.debug(f"  {asset}: shape={df.shape if hasattr(df, 'shape') else 'N/A'}")
            
            # Ensure all timeframes and assets are present in the processed data
            for tf in self.timeframes:
                if tf not in processed_data:
                    logger.warning(f"No data available for timeframe {tf}, creating empty data")
                    processed_data[tf] = {}
                
                # Ensure all assets have data for this timeframe
                for asset in self.assets:
                    if asset not in processed_data[tf]:
                        logger.warning(f"No data for {asset} {tf}, creating empty DataFrame")
                        empty_df = self._create_empty_dataframe(tf, window_size=self.window_sizes.get(tf, 1))
                        if empty_df is not None:
                            processed_data[tf][asset] = empty_df
                        else:
                            logger.error(f"Failed to create empty DataFrame for {asset} {tf}")
            
            if not processed_data:
                logger.error("No market data available to build observation.")
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            # Log final data structure before transformation
            logger.debug("=== Final Data Structure ===")
            for tf, assets in processed_data.items():
                logger.debug(f"Timeframe {tf} final assets: {list(assets.keys())}")
                for asset, df in assets.items():
                    logger.debug(f"  {asset}: shape={df.shape if hasattr(df, 'shape') else 'N/A'}")
            
            # Transform data structure from {timeframe: {asset: df}} to {asset: {timeframe: df}}
            transformed_data = {asset: {} for asset in self.assets}
            
            # Initialize all timeframes for all assets with empty DataFrames if needed
            for tf in self.timeframes:
                for asset in self.assets:
                    if tf not in processed_data or asset not in processed_data[tf]:
                        logger.warning(f"Creating empty DataFrame for {asset} {tf}")
                        empty_df = self._create_empty_dataframe(tf, window_size=self.window_sizes.get(tf, 1))
                        if empty_df is not None:
                            if tf not in processed_data:
                                processed_data[tf] = {}
                            processed_data[tf][asset] = empty_df
            
            # Now perform the transformation
            for tf in self.timeframes:
                if tf in processed_data:
                    for asset in self.assets:
                        if asset in processed_data[tf]:
                            transformed_data[asset][tf] = processed_data[tf][asset]
            
            # Log the transformed structure
            logger.debug("=== Transformed Data Structure ===")
            for asset, timeframes in transformed_data.items():
                logger.debug(f"Asset {asset} has data for timeframes: {list(timeframes.keys())}")
                for tf, df in timeframes.items():
                    logger.debug(f"  {tf}: shape={df.shape if hasattr(df, 'shape') else 'N/A'}")

            # Build the observation
            build_obs_start_time = time.time()
            observation = self.state_builder.build_adaptive_observation(
                self.current_step,
                transformed_data,  # Use the transformed data structure
                self.portfolio
            )
            build_obs_end_time = time.time()
            logger.debug(f"state_builder.build_adaptive_observation took {build_obs_end_time - build_obs_start_time:.4f} seconds")

            if not isinstance(observation, np.ndarray):
                logger.error(
                    f"Observation from build_adaptive_observation is not a numpy array: "
                    f"{type(observation)}"
                )
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            # Cache the observation
            self._manage_cache(cache_key, observation)
            return observation
            
        except Exception as e:
            logger.error(f"Error in _get_observation: {str(e)}", exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calcule la récompense pour l'étape actuelle."""
        portfolio_metrics = self.portfolio.get_metrics()
        returns = portfolio_metrics.get("returns", 0.0)
        max_drawdown = portfolio_metrics.get("max_drawdown", 0.0)
        reward_config = self.config.get("reward", {})
        return_scale = reward_config.get("return_scale", 1.0)
        risk_aversion = reward_config.get("risk_aversion", 1.5)
        base_reward = returns * return_scale
        risk_penalty = risk_aversion * max_drawdown
        transaction_penalty = 0.0
        if hasattr(self, "_last_portfolio_value"):
            turnover = abs(
                portfolio_metrics.get("total_value", 0.0)
                - self._last_portfolio_value
            ) / max(1.0, self._last_portfolio_value)
            transaction_penalty = (
                reward_config.get("transaction_cost_penalty", 0.1) * turnover
            )
        position_concentration = 0.0
        if portfolio_metrics.get("total_value", 0) > 0:
            position_values = [
                p.get("value", 0)
                for p in portfolio_metrics.get("positions", {}).values()
            ]
            if position_values:
                max_position = max(position_values)
                position_concentration = (
                    max_position / portfolio_metrics["total_value"]
                ) ** 2
        concentration_penalty = (
            reward_config.get("concentration_penalty", 0.5) * position_concentration
        )
        action_smoothness_penalty = 0.0
        if hasattr(self, "_last_action") and self._last_action is not None:
            action_diff = np.mean(np.abs(action - self._last_action))
            action_smoothness_penalty = (
                reward_config.get("action_smoothness_penalty", 0.1) * action_diff
            )
        reward = (
            base_reward
            - risk_penalty
            - transaction_penalty
            - concentration_penalty
            - action_smoothness_penalty
        )
        self._last_portfolio_value = portfolio_metrics.get("total_value", 0.0)
        self._last_action = action.copy()
        self._last_reward_components = {
            "base_reward": float(base_reward),
            "risk_penalty": float(risk_penalty),
            "transaction_penalty": float(transaction_penalty),
            "concentration_penalty": float(concentration_penalty),
            "action_smoothness_penalty": float(action_smoothness_penalty),
            "total_reward": float(reward),
        }
        return float(reward)

    def _get_info(self) -> Dict[str, Any]:
        """Récupère des informations supplémentaires sur l'état de l'environnement."""
        portfolio_metrics = self.portfolio.get_metrics()
        current_prices = self._get_current_prices()
        position_values = {}
        total_position_value = 0.0
        for asset, pos_info in portfolio_metrics.get("positions", {}).items():
            if asset in current_prices:
                qty = pos_info.get("quantity", 0)
                price = current_prices[asset]
                value = qty * price
                position_values[asset] = {
                    "quantity": qty,
                    "price": price,
                    "value": value,
                    "weight": (
                        value / portfolio_metrics.get("total_value", 1.0)
                        if portfolio_metrics.get("total_value", 0) > 0
                        else 0.0
                    ),
                }
                total_position_value += value
        reward_components = {}
        if hasattr(self, "_last_reward_components"):
            reward_components = self._last_reward_components
        action_stats = {}
        if hasattr(self, "_last_action") and self._last_action is not None:
            action = self._last_action
            action_stats = {
                "action_mean": float(np.mean(action)),
                "action_std": float(np.std(action)),
                "action_min": float(np.min(action)),
                "action_max": float(np.max(action)),
                "num_assets": len(action),
            }
        info = {
            "step": self.current_step,
            "chunk": self.current_chunk,
            "done": self.done,
            "portfolio": {
                "total_value": portfolio_metrics.get("total_value", 0.0),
                "cash": portfolio_metrics.get("cash", 0.0),
                "returns": portfolio_metrics.get("returns", 0.0),
                "max_drawdown": portfolio_metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", 0.0),
                "sortino_ratio": portfolio_metrics.get("sortino_ratio", 0.0),
                "total_position_value": total_position_value,
                "leverage": portfolio_metrics.get("leverage", 0.0),
                "num_positions": len(portfolio_metrics.get("positions", {})),
            },
            "positions": position_values,
            "market": {
                "num_assets": len(current_prices),
                "assets": list(current_prices.keys()),
                "current_prices": current_prices,
            },
            "action_stats": action_stats,
            "reward_components": reward_components,
            "performance": {
                "timestamp": self._get_safe_timestamp(),
                "steps_per_second": (
                    self.current_step
                    / max(0.0001, time.time() - self._episode_start_time)
                    if hasattr(self, "_episode_start_time")
                    else 0.0
                ),
            },
        }
        return info

    def render(self, mode: str = "human") -> None:
        """Affiche l'état actuel de l'environnement."""
        if mode == "human":
            portfolio_value = self.portfolio.get_portfolio_value()
            print(
                f"Étape: {self.current_step}, "
                f"Valeur du portefeuille: {portfolio_value:.2f}, "
                f"Espèces: {self.portfolio.cash:.2f}, "
                f"Positions: {self.portfolio.positions}"
            )

    def close(self) -> None:
        """Nettoie les ressources de l'environnement."""
        pass

