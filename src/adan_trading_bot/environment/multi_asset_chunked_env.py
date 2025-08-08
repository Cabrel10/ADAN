#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environnement de trading multi-actifs avec chargement par morceaux.

Ce module implémente un environnement de trading pour plusieurs actifs
avec chargement efficace des données par lots.
"""
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.data_processing.state_builder import StateBuilder, TimeframeConfig
from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
from adan_trading_bot.environment.order_manager import OrderManager
from adan_trading_bot.environment.reward_calculator import RewardCalculator
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager


# Configuration du logger
logger = logging.getLogger(__name__)


class MultiAssetChunkedEnv(gym.Env):
    """Environnement de trading multi-actifs avec chargement par morceaux.

    Cet environnement gère plusieurs actifs et intervalles de temps, avec
    support pour des espaces d'actions discrets et continus. Il utilise un
    constructeur d'état pour créer l'espace d'observation et un gestionnaire
    de portefeuille pour suivre les positions et le PnL.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

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

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.worker_config = worker_config
        self.data_loader_instance = data_loader_instance
        self.shared_buffer = shared_buffer

        # Initialize self.assets from worker_config
        self.assets = worker_config.get("assets", [])
        if not self.assets:
            raise ValueError("No assets specified in worker_config.")

        # Configuration du cache d'observations
        self._observation_cache = {}  # Cache des observations
        self._max_cache_size = 1000  # Taille maximale du cache
        self._cache_hits = 0  # Succès du cache
        self._cache_misses = 0  # Échecs du cache
        self._current_observation = None  # Observation courante
        self._cache_access = {}  # Suivi de l'utilisation

        # Initialisation des compteurs et états
        self.current_chunk = 0
        self.current_chunk_idx = 0
        self.done = False  # État done
        self.global_step = 0  # Compteur global d'étapes

        # Initialisation des composants critiques
        self._is_initialized = False  # Standardisation sur _is_initialized
        try:
            self._initialize_components()
            self._is_initialized = True
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation: %s", str(e))
            raise

    def _initialize_components(self) -> None:
        """Initialize all environment components in the correct order."""
        # 1. Initialize data loader FIRST to know the data structure
        if self.data_loader_instance is not None:
            self.data_loader = self.data_loader_instance
        else:
            self._init_data_loader(self.assets)

        # 2. Create TimeframeConfig from loaded data
        timeframe_configs = []
        if self.data_loader.features_by_timeframe:
            for tf_name, features in self.data_loader.features_by_timeframe.items():
                config = TimeframeConfig(
                    timeframe=tf_name, features=features, window_size=100
                )
                timeframe_configs.append(config)
        else:
            raise ValueError("No feature configuration in data loader.")

        # 3. Initialize portfolio manager
        portfolio_config = self.config.copy()
        env_config = self.config.get("environment", {})
        portfolio_config["trading_rules"] = self.config.get("trading_rules", {})
        portfolio_config["capital_tiers"] = self.config.get("capital_tiers", [])
        # Utiliser portfolio.initial_balance en priorité, puis environment.initial_balance
        # avec une valeur par défaut de 20.0
        portfolio_balance = self.config.get("portfolio", {}).get(
            "initial_balance", env_config.get("initial_balance", 20.0)
        )
        portfolio_config["initial_capital"] = portfolio_balance

        # Map asset names to full names (e.g., BTC -> BTCUSDT)
        asset_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
        }
        # Create a mapped assets list for the data loader
        mapped_assets = [asset_mapping.get(asset, asset) for asset in self.assets]

        # Initialize portfolio with mapped asset names
        self.portfolio = PortfolioManager(
            env_config=portfolio_config, assets=mapped_assets
        )
        # Create alias for backward compatibility
        self.portfolio_manager = self.portfolio
        self.assets = mapped_assets  # Update self.assets

        # Convert list of TimeframeConfig objects to dictionary
        timeframe_configs_dict = {
            tf_config.timeframe: tf_config for tf_config in timeframe_configs
        }

        # 4. Initialize StateBuilder with dynamic config
        features_config = {
            tf: config.features for tf, config in timeframe_configs_dict.items()
        }

        # Initialize with default target size first
        window_size = self.config.get("state", {}).get("max_window_size", 50)

        self.state_builder = StateBuilder(
            features_config=features_config,
            window_size=window_size,
            include_portfolio_state=True,
            normalize=True,
        )

        # 5. Setup action and observation spaces (requires state_builder)
        self._setup_spaces()

        # 6. Initialize other components using worker_config where available
        trading_rules = self.config.get("trading_rules", {})
        penalties = self.config.get("environment", {}).get("penalties", {})
        self.order_manager = OrderManager(
            trading_rules=trading_rules, penalties=penalties
        )

        # Get reward config with fallback to main config
        env_section = self.config.get("environment", {})
        reward_cfg = self.worker_config.get(
            "reward_config", env_section.get("reward_config", {})
        )

        # Create env config with reward shaping
        env_config = {"reward_shaping": reward_cfg}
        self.reward_calculator = RewardCalculator(env_config=env_config)

        # Initialize observation validator (will be initialized if needed)
        self.observation_validator = None

        # Initialize DBE with config from worker or main config
        dbe_config = self.worker_config.get("dbe_config", self.config.get("dbe", {}))
        self.dbe = DynamicBehaviorEngine(
            config=dbe_config,
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
            raise ValueError(
                "worker_config must be provided to initialize the data loader."
            )

        # Ensure paths are resolved
        if not hasattr(self, "config") or not self.config:
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
        mapped_assets = [asset_mapping.get(asset, asset) for asset in assets]

        if not mapped_assets:
            raise ValueError("No assets specified in worker or environment config")

        # Get timeframes from config with fallback to worker config
        global_data_timeframes = self.config.get("data", {}).get("timeframes", [])
        worker_timeframes = self.worker_config.get("timeframes", [])

        # Use worker timeframes if specified, otherwise fallback to global config
        self.timeframes = worker_timeframes or global_data_timeframes

        if not self.timeframes:
            raise ValueError(
                f"No timeframes defined: global={global_data_timeframes}, "
                f"worker={worker_timeframes}"
            )

        # Create a copy of the config with resolved paths
        loader_config = {
            **self.config,
            "data": {
                **self.config.get("data", {}),
                "features_per_timeframe": self.config.get("data", {}).get(
                    "features_per_timeframe", {}
                ),
                "assets": mapped_assets,  # Use mapped_assets here
            },
        }

        # Ensure we have features configured for each timeframe
        features_config = loader_config["data"]["features_per_timeframe"]
        for tf in self.timeframes:
            if tf not in features_config:
                logger.warning(
                    f"No features configured for timeframe {tf}, "
                    "using default features"
                )
                features_config[tf] = ["open", "high", "low", "close", "volume"]

        # Initialize the data loader
        self.data_loader = ChunkedDataLoader(
            config=loader_config,
            worker_config={
                **self.worker_config,
                "assets": mapped_assets,  # Use mapped_assets here
                "timeframes": self.timeframes,
            },
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

        # Configure observation space based on StateBuilder
        try:
            # Get the observation shape from StateBuilder (returns 3 values)
            (
                n_timeframes,
                window_size,
                n_features,
            ) = self.state_builder.get_observation_shape()

            # Get portfolio state dimension from StateBuilder
            portfolio_dim = self.state_builder.get_portfolio_state_dim()

            # Store the shape information
            self.observation_shape = (n_timeframes, window_size, n_features)
            self.portfolio_state_dim = portfolio_dim

            # Log the dimensions for debugging
            logger.info(f"Observation shape: {self.observation_shape}")
            logger.info(f"Portfolio state dimension: {self.portfolio_state_dim}")

            # Create observation space dictionary
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=self.observation_shape,
                        dtype=np.float32,
                    ),
                    "portfolio_state": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.portfolio_state_dim,),
                        dtype=np.float32,
                    ),
                }
            )

            logger.info(f"Observation space: {self.observation_space}")

        except Exception as e:
            logger.error(f"Error setting up observation space: {str(e)}")
            raise

    def reset(self, *, seed=None, options=None):
        """Reset the environment to start a new episode.

        Args:
            seed: Optional seed for the random number generator
            options: Additional options for reset

        Returns:
            tuple: (observation, info) containing the initial observation and info
        """
        super().reset(seed=seed)

        # Reset episode-specific variables
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.step_in_chunk = 0

        # Reset portfolio and order manager
        # Use new_epoch=True to ensure initial capital is properly set
        self.portfolio.reset(new_epoch=True)
        self.order_manager.reset()

        # Load initial chunk of data
        self.current_chunk_idx = 0
        self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)

        # Fit scalers on the first chunk of data
        if hasattr(self.state_builder, "fit_scalers"):
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
                        if dfs:  # Only concatenate if there are DataFrames
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

        # Get configuration
        state_config = self.config.get("state", {})
        env_config = self.config.get("environment", {})
        self.window_size = state_config.get("window_size", 50)
        self.warmup_steps = env_config.get("warmup_steps", self.window_size)

        # Log configuration
        logger.info(f"[CONFIG] Window size: {self.window_size}")
        logger.info(f"[CONFIG] Warmup steps: {self.warmup_steps}")

        # Get max steps configuration
        self.max_steps = env_config.get("max_steps", 1000)
        logger.info(f"[CONFIG] Max steps per episode: {self.max_steps}")

        # Initialize last trade step
        self.last_trade_step = 0
        logger.info("[INIT] Last trade step initialized to 0")

        if self.warmup_steps < self.window_size:
            msg = (
                f"warmup_steps ({self.warmup_steps}) is less than "
                f"window_size ({self.window_size}). Setting warmup_steps to "
                f"{self.window_size}"
            )
            logger.warning(msg)
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
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.current_step += 1
        self.global_step += 1

        # Log current step and action
        logger.info(f"[STEP {self.current_step}] Executing step with action: {action}")

        # Log portfolio value at the start of the step
        if hasattr(self, "portfolio_manager") and hasattr(
            self.portfolio_manager, "portfolio_value"
        ):
            logger.info(
                f"[STEP {self.current_step}] Portfolio value: {self.portfolio_manager.portfolio_value:.2f}"
            )
        else:
            logger.warning("[STEP] Portfolio manager or portfolio_value not available")

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
            logger.debug(
                f"_execute_trades took {trade_end_time - trade_start_time:.4f} seconds"
            )

            self.step_in_chunk += 1
            self.current_step += 1

            first_asset = next(iter(self.current_data.keys()))
            data_length = len(self.current_data[first_asset])

            MIN_EPISODE_STEPS = 500  # Minimum absolu avant évaluation
            done = False
            termination_reason = ""

            # Log current state before checking termination conditions
            logger.info(
                f"[TERMINATION CHECK] Step: {self.current_step}, "
                f"Max Steps: {self.max_steps}, "
                f"Portfolio Value: {self.portfolio_manager.portfolio_value:.2f}, "
                f"Initial Equity: {self.portfolio_manager.initial_equity:.2f}, "
                f"Steps Since Last Trade: {self.current_step - self.last_trade_step}"
            )

            if self.current_step < MIN_EPISODE_STEPS:
                done = False
                termination_reason = (
                    f"Min steps not reached ({self.current_step} < {MIN_EPISODE_STEPS})"
                )
                logger.info(f"[TERMINATION] {termination_reason}")
            elif self.current_step >= self.max_steps:
                done = True
                termination_reason = (
                    f"Max steps reached ({self.current_step} >= {self.max_steps})"
                )
                logger.info(f"[TERMINATION] {termination_reason}")
            elif self.portfolio_manager.total_value <= self.initial_equity * 0.70:
                done = True
                termination_reason = f"Max drawdown exceeded ({self.portfolio_manager.total_value:.2f} <= {self.initial_equity * 0.70:.2f})"
                logger.warning(f"[TERMINATION] {termination_reason}")
            elif self.current_step - self.last_trade_step > 300:
                done = True
                termination_reason = f"Max inactive steps reached ({self.current_step - self.last_trade_step} > 300)"
                logger.warning(f"[TERMINATION] {termination_reason}")
            elif self.current_step >= len(self.current_chunk) - 1:
                done = True
                termination_reason = f"End of chunk reached (step {self.current_step} >= {len(self.current_chunk) - 1})"
                logger.info(f"[TERMINATION] {termination_reason}")

            # Log final decision and handle episode termination
            if done:
                logger.info(
                    f"[EPISODE END] Episode ending. Reason: {termination_reason}"
                )
                logger.info(
                    f"[EPISODE STATS] Total steps: {self.current_step}, "
                    f"Final portfolio value: {self.portfolio_manager.total_value:.2f}, "
                    f"Return: {(self.portfolio_manager.total_value/self.initial_equity - 1)*100:.2f}%"
                )
            else:
                logger.debug(
                    f"[TERMINATION] Episode continues. Current step: {self.current_step}"
                )

                # Check for inactivity
                if (
                    self.current_step - self.last_trade_step > 300
                ):  # 300 steps without trading
                    termination_reason = f"Max inactive steps reached (Last trade: {self.current_step - self.last_trade_step} steps ago)"
                    logger.warning(f"[INACTIVITY] {termination_reason}")
                    done = True

                # Handle chunk transitions
                if self.step_in_chunk >= data_length:
                    logger.info(
                        f"[CHUNK] Reached end of chunk {self.current_chunk_idx + 1}/{self.total_chunks}"
                    )
                    self.current_chunk_idx += 1
                    self.current_chunk += 1
                    if self.current_chunk_idx >= self.total_chunks:
                        self.done = True
                        logger.info("[CHUNK] Reached final chunk, episode complete")
                    else:
                        logger.debug(
                            f"[CHUNK] Loading next chunk {self.current_chunk_idx + 1}/{self.total_chunks}"
                        )
                        self.current_data = self.data_loader.load_chunk(
                            self.current_chunk_idx
                        )
                        self.step_in_chunk = 0
                        if hasattr(self.dbe, "_reset_for_new_chunk"):
                            logger.debug("[DBE] Resetting DBE for new chunk")
                            self.dbe._reset_for_new_chunk()
                        logger.info(
                            f"[CHUNK] Successfully loaded chunk {self.current_chunk_idx + 1}"
                        )

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
                    "state": current_observation,
                    "action": action,
                    "reward": float(reward),
                    "next_state": next_observation,
                    "done": terminated or truncated,
                    "info": info,
                    "timestamp": self._get_safe_timestamp() or str(self.current_step),
                    "worker_id": self.worker_id,
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

            if hasattr(self, "current_data") and self.current_data:
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
            if hasattr(self, "dbe"):
                self.dbe.update_state(live_metrics)

        except Exception as e:
            logger.warning(f"Failed to update DBE state: {e}")

    def _execute_trades(
        self, action: np.ndarray, dbe_modulation: Dict[str, Any] = None
    ) -> None:
        """Execute trades based on the continuous action vector with improved validation and risk management.

        Args:
            action: Array of action values between -1 and 1 for each asset
            dbe_modulation: Optional dictionary of dynamic behavior engine parameters

        Raises:
            RuntimeError: If environment is not initialized
            ValueError: If action dimensions are invalid
        """
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        logger.info(f"[TRADE] Starting trade execution for step {self.current_step}")

        # Initialize trades executed counter
        trades_executed = 0

        # Validate action dimensions
        if len(action) != len(self.assets):
            error_msg = f"Action dimension {len(action)} does not match number of assets {len(self.assets)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Get current market data
            current_prices = self._get_current_prices()

            # Validate market data
            if not self._validate_market_data(current_prices):
                logger.error("Invalid market data, skipping trade execution")
                return

            # Get current portfolio state
            portfolio_metrics = self.portfolio.get_metrics()
            cash_before = portfolio_metrics.get("cash", 0.0)
            portfolio_value_before = self.portfolio.portfolio_value
            current_drawdown = self.portfolio.calculate_drawdown()
            current_leverage = self.portfolio.get_leverage()

            # Log current state
            logger.info(
                f"[PORTFOLIO] Before trades - "
                f"Value: ${portfolio_value_before:.2f}, "
                f"Cash: ${cash_before:.2f}, "
                f"Leverage: {current_leverage:.2f}x, "
                f"Drawdown: {current_drawdown*100:.2f}%"
            )

            # Log current prices and actions
            logger.info(
                f"[MARKET] Current prices: {', '.join([f'{a}: ${p:.2f}' for a, p in current_prices.items()])}"
            )
            logger.info(f"[ACTION] Raw actions: {dict(zip(self.assets, action))}")

            # Log open positions
            open_positions = [
                (asset, pos)
                for asset, pos in self.portfolio.positions.items()
                if hasattr(pos, "is_open") and pos.is_open
            ]

            if open_positions:
                logger.info("[TRADE] Current open positions:")
                for asset, pos in open_positions:
                    current_price = current_prices.get(asset, 0.0)
                    pnl_pct = (
                        ((current_price / pos.entry_price) - 1) * 100
                        if hasattr(pos, "entry_price") and pos.entry_price > 0
                        else 0
                    )
                    logger.info(
                        f"  - {asset}: {pos.size:.6f} @ ${pos.entry_price:.2f} "
                        f"(Current: ${current_price:.2f}, P&L: {pnl_pct:+.2f}%)"
                    )
            else:
                logger.info("[TRADE] No open positions")

            # Handle DBE modulation
            dbe_modulation = dbe_modulation or {}
            if dbe_modulation:
                logger.debug(f"DBE modulation: {dbe_modulation}")

            # Get trading parameters from config with defaults
            trading_config = self.config.get("trading", {})
            risk_config = trading_config.get("risk_management", {})

            base_sl = risk_config.get("base_stop_loss", 0.02)
            base_tp = risk_config.get("base_take_profit", 0.05)
            risk_per_trade = risk_config.get("risk_per_trade", 0.01)

            # Adjust risk based on drawdown (more conservative with higher drawdown)
            risk_multiplier = max(
                0.1, 1.0 - current_drawdown
            )  # Reduce risk as drawdown increases

            # Process each asset
            for i, (asset, action_value) in enumerate(zip(self.assets, action)):
                try:
                    action_value = float(np.clip(action_value, -1.0, 1.0))
                    price = current_prices.get(asset)

                    if price is None or price <= 0:
                        logger.warning(f"Invalid price {price} for {asset}, skipping")
                        continue

                    # Get or create position
                    position = self.portfolio.positions.get(asset)

                    # Determine action type
                    if action_value > 0.1:  # BUY
                        if position and position.is_open:
                            logger.debug(
                                f"[ORDER] Position in {asset} already open, holding"
                            )
                            continue

                        # Calculate position size with risk management
                        position_size = self.portfolio.calculate_position_size(
                            price=price,
                            risk_per_trade=risk_per_trade,
                            account_risk_multiplier=risk_multiplier,
                        )

                        if position_size > 0:
                            logger.info(
                                f"[ORDER] Opening position in {asset}: {position_size:.6f} @ ${price:.2f}"
                            )
                            self.order_manager.open_position(
                                portfolio=self.portfolio,
                                asset=asset,
                                price=price,
                                size=position_size,
                                stop_loss=price * (1 - base_sl),
                                take_profit=price * (1 + base_tp),
                            )
                            trades_executed += 1
                            self.last_trade_step = self.current_step

                    elif action_value < -0.1:  # SELL
                        if not position or not position.is_open:
                            logger.debug(f"[ORDER] No position in {asset} to close")
                            continue

                        logger.info(
                            f"[ORDER] Closing position in {asset} at ${price:.2f}"
                        )
                        self.order_manager.close_position(
                            portfolio=self.portfolio, asset=asset, price=price
                        )
                        trades_executed += 1
                        self.last_trade_step = self.current_step

                    # No action for values between -0.1 and 0.1 (HOLD)

                except Exception as e:
                    logger.error(
                        f"Error processing trade for {asset}: {str(e)}", exc_info=True
                    )

            # Update portfolio metrics after trades
            self.portfolio.update_market_price(current_prices)
            rebalance_result = self.portfolio.rebalance(current_prices)

            # Log portfolio state after trades
            portfolio_metrics = self.portfolio.get_metrics()
            cash_after = portfolio_metrics.get("cash", 0.0)
            portfolio_value_after = self.portfolio.portfolio_value

            logger.info(
                f"[PORTFOLIO] After trades - "
                f"Trades executed: {trades_executed}, "
                f"Cash: ${cash_after:.2f} (Δ${cash_after - cash_before:+.2f}), "
                f"Portfolio Value: ${portfolio_value_after:.2f} (Δ${portfolio_value_after - portfolio_value_before:+.2f})"
            )

            # Log detailed position information
            if hasattr(self.portfolio, "positions"):
                open_positions = [
                    asset
                    for asset, pos in self.portfolio.positions.items()
                    if hasattr(pos, "is_open") and pos.is_open
                ]
                if open_positions:
                    logger.info(
                        f"[POSITIONS] Open positions: {', '.join(open_positions)}"
                    )
                    for asset in open_positions:
                        pos = self.portfolio.positions[asset]
                        current_price = current_prices.get(asset, 0.0)
                        portfolio_metrics = self.portfolio.get_metrics()
                        position_info = (
                            f"  - {asset}: {pos.size:.4f} @ ${pos.entry_price:.2f} "
                            f"(Current: ${current_price:.2f}, "
                            f"P&L: ${portfolio_metrics.get('unrealized_pnl', 0):.2f}, "
                            f"Return: {portfolio_metrics.get('total_return_pct', 0):.2f}%)"
                        )
                        logger.info(position_info)
                else:
                    logger.info("[POSITIONS] No open positions")

        except Exception as e:
            logger.error(f"Critical error in _execute_trades: {str(e)}", exc_info=True)
            raise

    def _get_current_prices(self) -> Dict[str, float]:
        """Get the current prices for all assets with caching."""
        current_time = time.time()
        prices = {}

        # Vérifier si le cache est activé
        perf_config = self.config.get("trading", {}).get("performance", {})
        enable_caching = perf_config.get("enable_data_caching", True)

        for _asset, timeframe_data in self.current_data.items():
            # Vérifier si le prix est en cache et toujours valide
            cache_valid = (
                enable_caching
                and hasattr(self, "_price_cache")
                and _asset in self._price_cache
                and hasattr(self, "_last_price_update")
                and _asset in self._last_price_update
                and current_time - self._last_price_update[_asset] < self._cache_ttl
            )

            if cache_valid:
                prices[_asset] = self._price_cache[_asset]
                continue

            # Si pas en cache ou expiré, calculer le prix
            if not timeframe_data:
                continue

            tf = next(iter(timeframe_data.keys()))
            df = timeframe_data[tf]

            if not df.empty and self.current_step < len(df):
                try:
                    if "close" in df.columns:
                        price = float(df.iloc[self.current_step]["close"])
                        prices[_asset] = price
                    else:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            price = float(df.iloc[self.current_step][numeric_cols[0]])
                            prices[_asset] = price
                except Exception as e:
                    logger.warning(f"Error getting price for {_asset}: {str(e)}")

        return prices

    def _validate_market_data(self, prices: Dict[str, float]) -> bool:
        """Valide les données de marché avant l'exécution des trades.

        Args:
            prices: Dictionnaire des prix actuels par actif

        Returns:
            bool: True si les données sont valides, False sinon
        """
        if not prices:
            logger.error("No market data available")
            return False

        invalid_assets = [
            asset
            for asset, price in prices.items()
            if price <= 0 or not np.isfinite(price)
        ]

        if invalid_assets:
            invalid_list = ", ".join(invalid_assets)
            logger.error("Invalid prices for assets: %s", invalid_list)
            return False

        return True

    def _log_trade_error(
        self, asset: str, action_value: float, price: float, error: str
    ) -> None:
        """Enregistre les erreurs de trading pour analyse ultérieure.

        Args:
            asset: Symbole de l'actif concerné
            action_value: Valeur de l'action (-1 à 1)
            price: Prix au moment de l'erreur
            error: Message d'erreur détaillé
        """
        # Déterminer le type d'action
        if action_value > 0.1:
            action = "BUY"
        elif action_value < -0.1:
            action = "SELL"
        else:
            action = "HOLD"

        # Préparer les informations d'erreur
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "asset": asset,
            "action": action,
            "action_value": float(action_value),
            "price": float(price) if price is not None else None,
            "error": error,
        }

        # Ajouter la valeur du portefeuille si disponible
        if hasattr(self.portfolio, "portfolio_value"):
            error_info["portfolio_value"] = float(self.portfolio.portfolio_value)

        # Logger l'erreur
        logger.error(f"Trade error: {error_info}")

        # Enregistrer dans un fichier si configuré
        log_config = self.config.get("logging", {})
        error_log_path = log_config.get("error_log_path", "trade_errors.log")

        try:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_info) + "\n")
        except IOError as e:
            logger.error(f"Failed to write to error log: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error writing to log: {str(e)}")

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
                self._cache_access.keys(), key=lambda k: self._cache_access[k]
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
            "cache_enabled": True,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._observation_cache),
            "max_size": self._max_cache_size,
            "hit_ratio": hit_ratio,
        }

    def _process_assets(
        self, feature_config: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Process asset data for the current step.

        Args:
            feature_config: Dictionary mapping timeframes to lists of feature names

        Returns:
            Dictionary mapping assets to dictionaries of {timeframe: DataFrame}

            Format attendu par StateBuilder.build_observation():
            {
                'asset1': {
                    '5m': DataFrame,
                    '1h': DataFrame,
                    '4h': DataFrame
                },
                'asset2': {
                    '5m': DataFrame,
                    '1h': DataFrame,
                    '4h': DataFrame
                },
                ...
            }
        """
        processed_data = {asset: {} for asset in self.assets}

        for asset in self.assets:
            for timeframe in self.timeframes:
                if timeframe not in feature_config:
                    logger.warning(
                        f"No feature configuration for timeframe {timeframe}"
                    )
                    continue

                features = feature_config[timeframe]
                if not features:
                    logger.warning(f"No features specified for timeframe {timeframe}")
                    continue

                # Get data for this asset and timeframe
                asset_data = self.current_data.get(asset, {}).get(timeframe)
                if asset_data is None or asset_data.empty:
                    logger.debug(f"No data for {asset} {timeframe}")
                    continue

                # Log all available columns for debugging
                logger.debug(
                    f"Available columns in {asset} {timeframe} data: {asset_data.columns.tolist()}"
                )
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
                        logger.debug(
                            f"Found feature: '{f}' -> '{column_mapping[upper_f]}'"
                        )
                    else:
                        missing_features.append(f)
                        logger.debug(
                            f"Missing feature: '{f}' (not in DataFrame columns)"
                        )

                if missing_features:
                    logger.warning(
                        f"Missing {len(missing_features)} features for {asset} {timeframe}: {missing_features}"
                    )
                    logger.debug(
                        f"Available columns in {asset} {timeframe}: {asset_data.columns.tolist()}"
                    )
                    logger.debug(f"Available features: {available_features}")

                if not available_features:
                    logger.warning(
                        f"None of the requested features found for {asset} {timeframe}"
                    )
                    continue

                try:
                    # Select only the requested features using their original case
                    asset_df = asset_data[available_features].copy()

                    # Ensure column names are in uppercase for consistency
                    asset_df.columns = [col.upper() for col in asset_df.columns]

                    # Store the DataFrame in the processed data
                    processed_data[asset][timeframe] = asset_df

                except Exception as e:
                    logger.error(f"Error processing {asset} {timeframe}: {str(e)}")
                    logger.debug(f"Available columns: {asset_data.columns.tolist()}")
                    logger.debug(f"Available features: {available_features}")

        # Remove assets with no data
        return {k: v for k, v in processed_data.items() if v}

    def _create_empty_dataframe(
        self, timeframe: str, window_size: int = None
    ) -> pd.DataFrame:
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
                features = ["close"]  # Fallback to basic column

            # Use window_size if provided, otherwise default to 1
            rows = window_size if window_size is not None else 1

            # Create empty DataFrame with required features
            empty_data = np.zeros((rows, len(features)))
            df = pd.DataFrame(empty_data, columns=features)

            # Add timestamp column if not present
            if (
                "timestamp" not in df.columns
                and "timestamp" in self.features[timeframe]
            ):
                df["timestamp"] = pd.Timestamp.now()

            logger.debug(
                f"Created empty DataFrame for {timeframe} with shape {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error creating empty DataFrame for {timeframe}: {str(e)}")
            # Fallback to minimal DataFrame
            return pd.DataFrame(columns=["timestamp", "close"])

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construit et retourne l'observation actuelle sous forme de dictionnaire.

        Returns:
            Dict[str, np.ndarray]: Dictionnaire contenant :
                - 'observation': np.ndarray de forme (timeframes, window_size, features)
                - 'portfolio_state': np.ndarray de forme (17,) avec les métriques du portefeuille
        """
        try:
            # 1. Récupérer l'observation du marché depuis StateBuilder
            # Préparer les données pour build_observation
            data = self._process_assets(self.state_builder.features_config)
            market_obs = self.state_builder.build_observation(
                current_idx=self.current_step, data=data
            )

            # 2. Récupérer l'état du portefeuille
            portfolio_state = self.portfolio_manager.get_state()

            # 3. Vérifier et convertir les types
            market_obs = np.asarray(market_obs, dtype=np.float32)
            portfolio_state = np.asarray(portfolio_state, dtype=np.float32)

            # 4. Vérifier les dimensions
            if len(market_obs.shape) != 3:
                raise ValueError(
                    f"L'observation du marché doit être 3D, reçu : {market_obs.shape}"
                )

            if len(portfolio_state.shape) != 1 or portfolio_state.shape[0] != 17:
                raise ValueError(
                    f"L'état du portefeuille doit être de dimension (17,), reçu : {portfolio_state.shape}"
                )

            # 5. Créer et retourner le dictionnaire d'observation
            observation = {
                "observation": market_obs,
                "portfolio_state": portfolio_state,
            }

            # Mettre en cache l'observation actuelle
            self._current_observation = observation

            return observation

        except Exception as e:
            logger.error(f"Erreur dans _get_observation : {str(e)}")
            logger.error(traceback.format_exc())

            # En cas d'erreur, retourner une observation vide avec les bonnes dimensions
            num_timeframes = len(self.state_builder.features_config)
            # Utiliser max_features au lieu de la somme pour être cohérent avec StateBuilder
            num_features = (
                max(len(feats) for feats in self.state_builder.features_config.values())
                if self.state_builder.features_config
                else 1
            )
            return {
                "observation": np.zeros(
                    (num_timeframes, self.window_size, num_features), dtype=np.float32
                ),
                "portfolio_state": np.zeros(17, dtype=np.float32),
            }

    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calcule la récompense pour l'étape actuelle.

        La récompense est calculée comme suit :
        - Récompense de base : rendement du portefeuille
        - Pénalité de risque : basée sur le drawdown maximum
        - Pénalité de transaction : basée sur le turnover
        - Pénalité de concentration : pénalise les positions trop importantes
        - Pénalité de régularité des actions : pénalise les changements brusques

        Args:
            action: Vecteur d'actions du modèle

        Returns:
            float: Valeur de la récompense
        """
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Récupération des métriques du portefeuille
        portfolio_metrics = self.portfolio_manager.get_metrics()
        returns = portfolio_metrics.get("returns", 0.0)
        max_drawdown = portfolio_metrics.get("max_drawdown", 0.0)
        reward_config = self.config.get("reward", {})

        # Configuration des paramètres de récompense
        return_scale = reward_config.get("return_scale", 1.0)
        risk_aversion = reward_config.get("risk_aversion", 1.5)

        # Calcul de la récompense de base
        base_reward = returns * return_scale
        risk_penalty = risk_aversion * max_drawdown

        # Calcul de la pénalité de transaction
        transaction_penalty = 0.0
        if hasattr(self, "_last_portfolio_value"):
            last_value = self._last_portfolio_value
            current_value = portfolio_metrics.get("total_value", 0.0)
            turnover = abs(current_value - last_value) / max(1.0, last_value)
            transaction_penalty = (
                reward_config.get("transaction_cost_penalty", 0.1) * turnover
            )

        # Calcul de la pénalité de concentration
        position_concentration = 0.0
        if portfolio_metrics.get("total_value", 0) > 0:
            positions = portfolio_metrics.get("positions", {})
            position_values = [p.get("value", 0) for p in positions.values()]
            if position_values:
                max_position = max(position_values)
                position_concentration = (
                    max_position / portfolio_metrics["total_value"]
                ) ** 2

        concentration_penalty = (
            reward_config.get("concentration_penalty", 0.5) * position_concentration
        )

        # Calcul de la pénalité de régularité des actions
        action_smoothness_penalty = 0.0
        if hasattr(self, "_last_action") and self._last_action is not None:
            action_diff = np.mean(np.abs(action - self._last_action))
            action_smoothness_penalty = (
                reward_config.get("action_smoothness_penalty", 0.1) * action_diff
            )

        # Calcul de la récompense totale
        reward = (
            base_reward
            - risk_penalty
            - transaction_penalty
            - concentration_penalty
            - action_smoothness_penalty
        )

        # Mise à jour des états pour la prochaine itération
        self._last_portfolio_value = portfolio_metrics.get("total_value", 0.0)
        self._last_action = action.copy()

        # Journalisation des composantes de la récompense
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
        """
        Récupère des informations supplémentaires sur l'état de l'environnement.

        Returns:
            Dict[str, Any]: Dictionnaire contenant des informations détaillées sur l'état
                actuel du portefeuille et de l'environnement.
        """
        portfolio_metrics = self.portfolio_manager.get_metrics()
        current_prices = self._get_current_prices()
        position_values = {}
        total_position_value = 0.0

        # Calculer les valeurs des positions actuelles
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
