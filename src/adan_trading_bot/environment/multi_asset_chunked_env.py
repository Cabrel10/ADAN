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

        # Ajout d'un cache d'observations avec une taille maximale
        self._observation_cache = {}
        self._max_cache_size = 1000  # Nombre maximum d'observations en cache
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_access = {} # Pour suivre la dernière utilisation

        # Initialisation des composants critiques
        self._initialized = False
        try:
            self._initialize_components()
            self._setup_spaces()
            self._initialized = True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {str(e)}")
            raise

    def _initialize_components(self) -> None:
        """Initialize all environment components in the correct order."""
        # 1. Initialize data loader FIRST to know the data structure
        if not hasattr(self, 'data_loader') or self.data_loader is None:
            if self.data_loader_instance is not None:
                self.data_loader = self.data_loader_instance
            else:
                self._init_data_loader()

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
        portfolio_config["assets"] = self.assets
        self.portfolio = PortfolioManager(env_config=portfolio_config)

        # Convert list of TimeframeConfig objects to dictionary
        timeframe_configs_dict = {
            tf_config.timeframe: tf_config for tf_config in timeframe_configs
        }

        # 4. Initialize StateBuilder with the dynamic config
        # Convert timeframe_configs_dict to features_config format expected by StateBuilder
        features_config = {
            tf: config.features
            for tf, config in timeframe_configs_dict.items()
        }

        self.state_builder = StateBuilder(
            features_config=features_config,
            window_size=self.config.get("state", {}).get("window_size", 50),
            include_portfolio_state=True,
            normalize=True
        )

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

    def _init_data_loader(self) -> Any:
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
        # Get assets from worker config or environment
        self.assets = self.worker_config.get("assets", self.config.get("environment", {}).get("assets", []))

        if not self.assets:
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

        # Ensure we have features configured for each timeframe
        features_config = self.config.get("data", {}).get("features_per_timeframe", {})
        for tf in self.timeframes:
            if tf not in features_config:
                logger.warning(f"No features configured for timeframe {tf}, using default features")
                features_config[tf] = ["open", "high", "low", "close", "volume"]

        # Create a copy of the config with resolved paths
        loader_config = {
            **self.config,
            "data": {
                **self.config.get("data", {}),
                "features_per_timeframe": features_config,
                "assets": self.assets
            }
        }

        # Initialize the data loader
        self.data_loader = ChunkedDataLoader(
            config=loader_config,
            worker_config={
                **self.worker_config,
                "assets": self.assets,
                "timeframes": self.timeframes
            }
        )

        # Store assets list for easy access
        self.assets = self.data_loader.assets_list
        if not self.assets:
            raise ValueError("No assets available for trading after data loader initialization")

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
            # Get the 3D shape from StateBuilder
            shape_3d = self.state_builder.get_observation_shape()
            if len(shape_3d) != 3:
                raise ValueError(f"Expected 3D shape from StateBuilder, got {shape_3d}")

            n_timeframes, window_size, n_features = shape_3d

            # Validate dimensions
            if n_timeframes <= 0 or window_size <= 0 or n_features <= 0:
                raise ValueError(f"Invalid dimensions in observation shape: {shape_3d}")

            # Calculate total flattened dimension
            total_obs_size = n_timeframes * window_size * n_features

            logger.info(
                f"Observation space shape: {n_timeframes} timeframes * "
                f"{window_size} steps * {n_features} features = "
                f"{total_obs_size} total features"
            )

            # Create a flat Box space that matches _get_observation() output
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_obs_size,),
                dtype=np.float32,
            )

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

        # Reset state
        self.current_step = 0
        self._last_observation = None
        self.current_chunk = 0
        self.done = False
        self._observation_cache = cache_backup
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_action = None
        self._last_reward_components = {}
        self._episode_start_time = time.time()

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
        
        # Aplatir l'observation si nécessaire
        if len(observation.shape) > 1:
            observation = observation.flatten()
            
        # Vérifier que la forme est correcte
        if observation.shape != self.observation_space.shape:
            try:
                # Essayer de redimensionner pour correspondre à l'espace d'observation
                observation = np.resize(observation, self.observation_space.shape)
                logger.warning(
                    f"Observation shape {observation.shape} resized to match "
                    f"observation space shape {self.observation_space.shape}"
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to reshape observation from {observation.shape} to "
                    f"{self.observation_space.shape}: {str(e)}"
                )

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        self._last_action = action

        if self.done:
            logger.warning(
                "Calling step() after the episode has ended. "
                "You should call reset() before stepping again."
            )
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, True, False, info

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
            self._execute_trades(action, dbe_modulation)

            self.step_in_chunk += 1
            self.current_step += 1

            first_asset = next(iter(self.current_data.keys()))
            data_length = len(self.current_data[first_asset])

            if self.step_in_chunk >= data_length:
                self.current_chunk_idx += 1
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
            
            # Aplatir l'observation si nécessaire
            if len(next_observation.shape) > 1:
                next_observation = next_observation.flatten()
                
            # Vérifier que la forme est correcte
            if next_observation.shape != self.observation_space.shape:
                try:
                    # Essayer de redimensionner pour correspondre à l'espace d'observation
                    next_observation = np.resize(next_observation, self.observation_space.shape)
                    logger.warning(
                        f"Observation shape {next_observation.shape} resized to match "
                        f"observation space shape {self.observation_space.shape}"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to reshape observation from {next_observation.shape} to "
                        f"{self.observation_space.shape}: {str(e)}"
                    )

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
            current_position = portfolio_metrics["positions"].get(
                asset, {"quantity": 0.0}
            )
            current_qty = current_position.get("quantity", 0.0)
            current_value = current_qty * current_prices[asset]

            if action_value >= 0:
                if action_value == 0:
                    target_value = current_value
                else:
                    max_buy_value = cash * action_value
                    target_value = current_value + max_buy_value
            else:
                sell_pct = abs(action_value)
                target_value = current_value * (1 - sell_pct)

            if action_value > 0.1:
                discrete_action = 1
            elif action_value < -0.1:
                discrete_action = 2
            else:
                discrete_action = 0

            if discrete_action == 1: # BUY
                self.order_manager.open_position(self.portfolio, asset, current_prices[asset])
            elif discrete_action == 2: # SELL
                self.order_manager.close_position(self.portfolio, asset, current_prices[asset])
            cash -= target_value - current_value

        self.portfolio.update_market_price(current_prices)
        self.portfolio.rebalance(current_prices)

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all assets."""
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

    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the environment."""
        cache_key = f"{self.current_chunk_idx}_{self.step_in_chunk}_{self.current_step}"
        cached_obs = self._manage_cache(cache_key)
        if cached_obs is not None:
            return cached_obs

        try:
            feature_config = {
                tf: self.state_builder.get_feature_names(tf)
                for tf in self.timeframes
            }
            if not feature_config:
                raise ValueError("Empty feature configuration from StateBuilder")

            all_assets_features = self._process_assets(feature_config)
            
            if not all_assets_features:
                logger.error("No market data available to build observation.")
                return np.full(self.observation_space.shape, np.nan, dtype=np.float32)

            combined_data = pd.concat(all_assets_features, axis=0)

            if combined_data.empty:
                logger.error("Combined data is empty after processing.")
                return np.full(self.observation_space.shape, np.nan, dtype=np.float32)

            observation = self._build_observation(combined_data, cache_key)
            if not isinstance(observation, np.ndarray):
                logger.error(f"Observation from _build_observation is not a numpy array: {type(observation)}")
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation

        except Exception as e:
            logger.error("Error building observation: %s", str(e), exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _process_assets(self, feature_config: Dict[str, List[str]]) -> List[pd.DataFrame]:
        """Traite les actifs pour extraire les caractéristiques nécessaires."""
        all_assets_features = []
        
        for asset in self.assets:
            if asset not in self.current_data:
                logger.warning("No data for asset %s, skipping.", asset)
                continue
                
            asset_timeframe_data = self.current_data[asset]
            asset_features = {}
            
            for tf_name, required_features in feature_config.items():
                if tf_name not in asset_timeframe_data:
                    logger.warning(
                        "Timeframe %s not found for asset %s. Available: %s",
                        tf_name, asset, list(asset_timeframe_data.keys())
                    )
                    continue
                    
                df = asset_timeframe_data[tf_name]
                if df is None or df.empty:
                    logger.warning("DataFrame is empty for %s %s", asset, tf_name)
                    continue
                    
                if self.step_in_chunk >= len(df):
                    logger.warning(
                        "Step %s is out of bounds for %s %s (length: %s)",
                        self.step_in_chunk, asset, tf_name, len(df)
                    )
                    continue

                current_row = df.iloc[self.step_in_chunk]
                available_features = [
                    f for f in required_features if f in current_row.index
                ]
                
                if not available_features:
                    logger.warning("No available features for %s in %s", asset, tf_name)
                    continue

                for feature in available_features:
                    asset_features[f"{tf_name}_{feature}"] = current_row[feature]

            if asset_features:
                asset_df = pd.Series(asset_features, name=asset).to_frame().T
                all_assets_features.append(asset_df)
            else:
                logger.warning("No features available for asset %s", asset)
                
        return all_assets_features

    def _build_observation(self, combined_data: pd.DataFrame, cache_key: str) -> np.ndarray:
        """Construit l'observation à partir des données combinées."""
        timeframe_data = {}
        for tf in self.timeframes:
            tf_columns = [
                col for col in combined_data.columns
                if col.startswith(f"{tf}_")
            ]
            if tf_columns:
                tf_data = combined_data[tf_columns].copy()
                tf_data.columns = [
                    col.replace(f"{tf}_", "")
                    for col in tf_data.columns
                ]
                timeframe_data[tf] = tf_data

        current_idx = max(self.current_step, self.window_size)
        
        # Récupérer l'observation adaptative 3D
        observation = self.state_builder.build_adaptive_observation(
            current_idx, timeframe_data
        )
        
        if observation is None:
            logger.error("Échec de la construction de l'observation adaptative")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        self._manage_cache(cache_key, observation)

        # Aplatir l'observation 3D en 1D pour correspondre à l'espace d'observation
        observation = observation.reshape(-1)  # Aplatir en 1D
        
        # Vérifier la taille de l'observation
        expected_size = np.prod(self.observation_space.shape)
        if observation.size != expected_size:
            logger.warning(
                f"Taille d'observation incorrecte : {observation.size} vs attendu {expected_size}. "
                "Redimensionnement en cours..."
            )
            # Créer un nouveau tableau de la taille attendue
            resized_obs = np.zeros(expected_size, dtype=observation.dtype)
            # Copier les données disponibles
            min_size = min(observation.size, expected_size)
            resized_obs[:min_size] = observation.flat[:min_size]
            observation = resized_obs

        # Gérer les valeurs manquantes ou infinies
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            logger.warning(
                "NaN/Inf détectés dans l'observation: %s NaN, %s Inf. Remplacement par des zéros.",
                np.isnan(observation).sum(),
                np.isinf(observation).sum()
            )
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        return observation

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