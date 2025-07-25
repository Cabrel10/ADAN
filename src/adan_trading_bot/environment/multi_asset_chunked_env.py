#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-asset trading environment with chunked data loading.

This environment extends the base trading environment to support multiple
assets and efficient data loading using ChunkedDataLoader.
"""

import logging
import time
import numpy as np
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import pandas as pd
from gymnasium import spaces

from ..data_processing.chunked_loader import ChunkedDataLoader
from ..data_processing.observation_validator import ObservationValidator
from ..data_processing.state_builder import StateBuilder, TimeframeConfig
from ..environment.dynamic_behavior_engine import DynamicBehaviorEngine
from ..portfolio.portfolio_manager import PortfolioManager
from ..trading.order_manager import OrderManager

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Simple reward calculator for the environment."""

    def __init__(
        self,
        return_scale: float = 1.0,
        risk_free_rate: float = 0.0,
        max_drawdown_penalty: float = 0.0,
    ):
        """Initialize the reward calculator."""
        self.return_scale = return_scale
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty

    def calculate(
        self,
        returns: float,
        risk_free_rate: float = 0.0,
        max_drawdown: float = 0.0,
    ) -> float:
        """Calculate reward based on returns and risk metrics."""
        reward = returns * self.return_scale
        if max_drawdown > 0:
            reward -= max_drawdown * self.max_drawdown_penalty
        return reward


class MultiAssetChunkedEnv(gym.Env):
    """Multi-asset trading environment with chunked data loading.

    This environment loads data in chunks to manage memory usage and supports
    trading multiple assets simultaneously.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(
        self,
        config: Dict[str, Any],
        data_loader_instance: Optional[Any] = None
    ):
        """
        Initialize the multi-asset trading environment.

        Args:
            config: Configuration dictionary containing:
                - data: Data loading configuration
                - environment: Environment parameters
                - portfolio: Portfolio management settings
                - trading: Trading parameters
            data_loader_instance: Pre-initialized data loader for testing.
        """
        super().__init__()
        self.config = config
        # Initialize self.assets and timeframes safely
        self.assets = self.config.get("data", {}).get("assets", [])
        self.timeframes = self.config.get("data", {}).get("timeframes", ["5m", "1h", "4h"])
        self._validate_config()
        # Store the instance
        self.data_loader_instance = data_loader_instance
        self._initialize_components()

        # Initialize state
        self.current_step = 0
        self.current_chunk = 0
        self.current_data = {}
        self.done = False
        self._observation_cache = {}  # Initialize observation cache

        # Load the first chunk of data to determine observation space
        self.current_data = self.data_loader.load_chunk(0)
        self._setup_spaces()

        logger.info("MultiAssetChunkedEnv initialized")

    def get_applicable_tier(self, capital: float) -> Dict[str, Any]:
        """
        Get the capital tier applicable to the given capital amount.
        
        Args:
            capital: The current capital amount.
            
        Returns:
            The applicable tier configuration dictionary.
            
        Raises:
            ValueError: If no applicable tier is found.
        """
        if not hasattr(self, 'config') or 'capital_tiers' not in self.config:
            raise ValueError("Capital tiers not configured")
            
        # Sort tiers by min_capital to ensure correct order
        tiers = sorted(self.config["capital_tiers"], key=lambda x: x["min_capital"])
        
        for tier in tiers:
            min_cap = tier["min_capital"]
            max_cap = tier["max_capital"]
            
            # Check if capital falls within this tier's range
            if (capital >= min_cap and 
                (max_cap is None or capital < max_cap)):
                return tier
        
        # If we get here, no tier was found - use the highest tier as fallback
        if tiers:
            return tiers[-1]
            
        raise ValueError("No capital tiers defined")
        
    def _validate_config(self) -> None:
        """
        Validate the configuration dictionary.
        
        Raises:
            ValueError: If any required section or field is missing or invalid.
        """
        # Check required top-level sections
        required_sections = ["data", "environment", "portfolio", "trading_rules", "capital_tiers"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")
        
        # Validate capital_tiers structure
        required_tier_fields = [
            "name", "min_capital", "max_capital", "max_position_size_pct",
            "leverage", "risk_per_trade_pct", "max_drawdown_pct"
        ]
        
        if not isinstance(self.config["capital_tiers"], list):
            raise ValueError("capital_tiers must be a list of tier configurations")
            
        for i, tier in enumerate(self.config["capital_tiers"]):
            if not isinstance(tier, dict):
                raise ValueError(f"Tier at index {i} is not a dictionary")
                
            for field in required_tier_fields:
                if field not in tier:
                    raise ValueError(f"Tier at index {i} is missing required field: {field}")
            
            # Validate min_capital and max_capital
            if not (isinstance(tier["min_capital"], (int, float)) and tier["min_capital"] >= 0):
                raise ValueError(f"Invalid min_capital in tier {i}: {tier['min_capital']}")
                
            if tier["max_capital"] is not None and not (isinstance(tier["max_capital"], (int, float)) and tier["max_capital"] > 0):
                raise ValueError(f"Invalid max_capital in tier {i}: {tier['max_capital']}")
                
            # Validate percentages (0-100)
            for pct_field in ["max_position_size_pct", "risk_per_trade_pct", "max_drawdown_pct"]:
                if not (0 <= tier[pct_field] <= 100):
                    raise ValueError(f"{pct_field} must be between 0 and 100 in tier {i}")
            
            # Validate leverage (>= 1.0)
            if not (isinstance(tier["leverage"], (int, float)) and tier["leverage"] >= 1.0):
                raise ValueError(f"leverage must be >= 1.0 in tier {i}")
        
        # Check tier ordering (min_capital should be increasing)
        tiers = sorted(self.config["capital_tiers"], key=lambda x: x["min_capital"])
        for i in range(1, len(tiers)):
            if tiers[i]["min_capital"] <= tiers[i-1]["min_capital"]:
                raise ValueError("Tiers must be ordered by increasing min_capital")
            
            # Update the config with sorted tiers
            self.config["capital_tiers"] = tiers

    def _initialize_components(self) -> None:
        """Initialize all environment components in the correct order."""
        # 1. Initialize data loader FIRST to know the data structure
        if self.data_loader_instance:
            self.data_loader = self.data_loader_instance
        else:
            self._init_data_loader()

        # 2. Dynamically create TimeframeConfig from the loaded data
        timeframe_configs = []
        if self.data_loader.features_by_timeframe:
            for tf_name, features in self.data_loader.features_by_timeframe.items():
                timeframe_configs.append(
                    TimeframeConfig(name=tf_name, features=features, weight=1.0)
                )
        else:
            raise ValueError("No feature configuration found in data loader.")

        # 3. Initialize portfolio manager
        portfolio_config = self.config.copy()
        env_config = self.config.get('environment', {})
        portfolio_config["trading_rules"] = self.config.get("trading_rules", {})
        portfolio_config["capital_tiers"] = self.config.get("capital_tiers", [])
        portfolio_config["initial_capital"] = env_config.get("initial_capital", 10000.0)
        portfolio_config["assets"] = self.assets
        self.portfolio = PortfolioManager(env_config=portfolio_config)

        # Convert list of TimeframeConfig objects to dictionary
        timeframe_configs_dict = {tf_config.name: tf_config for tf_config in timeframe_configs}
        
        # 4. Initialize StateBuilder with the dynamic config
        self.state_builder = StateBuilder(
            config=self.config,
            assets=self.assets,
            timeframes=self.timeframes,
            timeframe_configs=timeframe_configs_dict,
            portfolio_manager=self.portfolio,
            window_size=self.config.get("state", {}).get("window_size", 50)
        )

        # 5. Initialize other components
        self.order_manager = OrderManager(portfolio_manager=self.portfolio)
        env_cfg = self.config.get("environment", {})
        self.reward_calculator = RewardCalculator(
            return_scale=env_cfg.get("return_scale", 1.0),
            risk_free_rate=env_cfg.get("risk_free_rate", 0.0),
            max_drawdown_penalty=env_cfg.get("max_drawdown_penalty", 0.0),
        )
        self.observation_validator = ObservationValidator()
        dbe_config = self.config.get("dbe", {})
        self.dbe = DynamicBehaviorEngine(
            config=dbe_config,
            finance_manager=getattr(self.portfolio, "finance_manager", None),
        )

    def _init_data_loader(self) -> None:
        """Initialize the chunked data loader."""
        data_config = self.config["data"]

        self.data_loader = ChunkedDataLoader(config=self.config)

        # Store assets list for easy access
        self.assets = self.data_loader.assets_list
        if not self.assets:
            raise ValueError("No assets available for trading")

        logger.info(f"Initialized data loader with {len(self.assets)} assets")

    def _get_features_config(self) -> Dict[str, List[str]]:
        """Extract features configuration from the main config."""
        features_config = {}

        if (
            "feature_engineering" in self.config
            and "timeframes" in self.config["feature_engineering"]
        ):
            for tf in self.config["feature_engineering"]["timeframes"]:
                features_config[tf] = (
                    self.config["feature_engineering"]
                    .get("features", {})
                    .get(tf, [
                        "open", "high", "low", "close",
                        "volume", "minutes_since_update"
                    ])
                )
        else:
            # Default features if not specified
            default_features = [
                "open", "high", "low", "close",
                "volume", "minutes_since_update"
            ]
            features_config = {
                "5m": default_features.copy(),
                "1h": default_features.copy(),
                "4h": default_features.copy(),
            }

        return features_config

    def _validate_observation_shape(self, observation: np.ndarray) -> None:
        """Validate that the observation shape matches the expected shape.

        Args:
            observation: The observation array to validate

        Raises:
            ValueError: If the observation shape is invalid
        """
        if not isinstance(observation, np.ndarray):
            raise ValueError(
                f"Observation must be a numpy array, got {type(observation)}"
            )

        expected_shape = self.observation_space.shape
        if observation.shape != expected_shape:
            raise ValueError(
                f"Observation shape {observation.shape} does not match "
                f"expected shape {expected_shape}"
            )

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces.

        Raises:
            ValueError: If the observation space cannot be properly configured
        """
        # Action space: Continuous actions in [-1, 1] for each asset
        # -1 = max sell, 0 = hold, 1 = max buy
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.assets),),  # One action per asset
            dtype=np.float32
        )

        try:
            # Get the 3D shape from StateBuilder
            shape_3d = self.state_builder.get_observation_shape()
            if len(shape_3d) != 3:
                raise ValueError(
                    f"Expected 3D shape from StateBuilder, got {shape_3d}"
                )
                
            n_timeframes, window_size, n_features = shape_3d
            
            # Validate dimensions
            if n_timeframes <= 0 or window_size <= 0 or n_features <= 0:
                raise ValueError(
                    f"Invalid dimensions in observation shape: {shape_3d}"
                )

            # Calculate total flattened dimension
            total_obs_size = n_timeframes * window_size * n_features
            
            logger.info(
                f"Observation space shape: {n_timeframes} timeframes * "
                f"{window_size} steps * {n_features} features = "
                f"{total_obs_size} total features"
            )
            
            # Create a flat Box space that matches _get_observation() output
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_obs_size,),
                dtype=np.float32,
            )
            
            # Verify the StateBuilder can produce an observation of the expected size
            try:
                test_obs = self.state_builder.build_observation(
                    pd.DataFrame(columns=self.state_builder.get_feature_names())
                )
                if len(test_obs) != total_obs_size:
                    logger.warning(
                        f"StateBuilder produced observation of size {len(test_obs)} "
                        f"but expected {total_obs_size}"
                    )
            except Exception as e:
                logger.warning(f"Failed to validate StateBuilder: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to set up observation space: {str(e)}")
            raise

    def reset(self, *, seed=None, options=None):
        # Reset environment state
        self.current_step = 0
        self.current_chunk = 0
        self.done = False
        self._observation_cache = {}  # Clear observation cache on reset
        self._last_action = None  # Reset last action
        self._last_reward_components = {}  # Reset reward components
        self._episode_start_time = time.time()  # Track episode start time

        # Reset portfolio
        self.portfolio.reset()

        # Load the first chunk of data
        self.current_chunk_idx = 0
        self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)

        # --- IMPROVED WARM-UP LOGIC ---
        # Get window_size and warmup_steps from config
        state_config = self.config.get('state', {})
        env_config = self.config.get('environment', {})
        self.window_size = state_config.get('window_size', 50)
        self.warmup_steps = env_config.get('warmup_steps', self.window_size)

        # Ensure warmup_steps is at least window_size
        if self.warmup_steps < self.window_size:
            logger.warning(
                f"warmup_steps ({self.warmup_steps}) is less than "
                f"window_size ({self.window_size}). Setting warmup_steps to "
                f"{self.window_size}"
            )
            self.warmup_steps = self.window_size

        # Get the length of the first DataFrame in the current chunk
        first_asset = next(iter(self.current_data.keys()))
        first_timeframe = next(iter(self.current_data[first_asset].keys()))
        data_length = len(self.current_data[first_asset][first_timeframe])

        logger.info(
            f"Reset: current_step={self.current_step}, "
            f"warmup_steps={self.warmup_steps}, "
            f"window_size={self.window_size}, data_length={data_length}"
        )

        if data_length < self.warmup_steps:
            raise ValueError(
                f"Le premier chunk ({data_length} steps) est plus petit "
                f"que la période de warm-up requise "
                f"({self.warmup_steps} steps)."
            )

        # Position ourselves at the start of the warm-up period
        self.step_in_chunk = 0

        # Skip the warm-up period
        for _ in range(self.warmup_steps - 1):
            self.step_in_chunk += 1
            self.current_step += 1 # Increment global step during warm-up
            if self.step_in_chunk >= data_length:
                # If we reach the end of the chunk during warm-up,
                # load the next chunk
                self.current_chunk_idx += 1
                if self.current_chunk_idx >= self.data_loader.total_chunks:
                    raise ValueError(
                        "Reached end of data during warm-up period. "
                        f"Current chunk: {self.current_chunk_idx}, "
                        f"Total chunks: {self.data_loader.total_chunks}"
                    )
                self.current_data = self.data_loader.load_chunk(
                    self.current_chunk_idx
                )
                self.step_in_chunk = 0
                first_asset = next(iter(self.current_data.keys())) # Corrected access
                first_timeframe = next(iter(self.current_data[first_asset].keys())) # Corrected access
                data_length = len(self.current_data[first_asset][first_timeframe])

        logger.info(
            f"Environment reset. "
            f"Warm-up completed, starting at step {self.step_in_chunk}."
        )

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        # Ensure observation is a flat numpy array
        observation = np.asarray(observation, dtype=np.float32).flatten()

        # Verify observation shape matches the defined observation space
        if observation.shape != self.observation_space.shape:
            raise ValueError(
                f"Observation shape {observation.shape} does not match "
                f"observation space shape {self.observation_space.shape}"
            )

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment.

        Args:
            action: Action array with values in [-1, 1] for each asset
                -1 = max sell, 0 = hold, 1 = max buy

        Returns:
            observation: Next observation as a flat numpy array
            reward: Reward for the current step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Store the last action for info and potential use
        # in reward calculation
        self._last_action = action

        # Check if the episode is already done
        if self.done:
            logger.warning(
                "Calling step() after the episode has ended. "
                "You should call reset() before stepping again."
            )
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, True, False, info

        try:
            # Ensure action is a numpy array and clip to valid range
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

            # Validate action shape
            if action.shape != (len(self.assets),):
                raise ValueError(
                    f"Action shape {action.shape} does not match "
                    f"expected shape (n_assets={len(self.assets)},)"
                )

            # Update DBE with current market conditions
            self._update_dbe_state()

            # Get dynamic modulation from DBE
            dbe_modulation = self.dbe.compute_dynamic_modulation()

            # Execute trades based on actions (with DBE modulation)
            self._execute_trades(action, dbe_modulation)

            # Advance step counters
            self.step_in_chunk += 1
            self.current_step += 1  # Global step counter

            # Check if we need to load the next chunk
            first_asset = next(iter(self.current_data.keys()))
            data_length = len(self.current_data[first_asset])

            if self.step_in_chunk >= data_length:
                self.current_chunk_idx += 1
                if self.current_chunk_idx >= self.data_loader.total_chunks:
                    self.done = True
                else:
                    self.current_data = self.data_loader.load_chunk(
                        self.current_chunk_idx
                    )
                    self.step_in_chunk = 0  # Reset for new chunk
                    # Reset DBE state for new chunk if needed
                    if hasattr(self.dbe, '_reset_for_new_chunk'):
                        self.dbe._reset_for_new_chunk()

            # Get next observation
            observation = self._get_observation()

            # Ensure observation is a flat numpy array
            observation = np.asarray(observation, dtype=np.float32).flatten()

            # Verify observation shape matches the defined observation space
            if observation.shape != self.observation_space.shape:
                raise ValueError(
                    f"Observation shape {observation.shape} does not match "
                    f"observation space shape {self.observation_space.shape}"
                )

            # Calculate reward
            reward = self._calculate_reward(action)

            # Check if episode is done
            terminated = self.done
            truncated = False  # Can be set by wrappers or timeout

            # Check for infinite loop or max steps
            max_steps = getattr(self, '_max_episode_steps', float('inf'))
            if self.current_step >= max_steps:
                truncated = True
                self.done = True

            # Get additional info
            info = self._get_info()

            # Store reward components for info
            if hasattr(self, '_last_reward_components'):
                info.update(
                    {"reward_components": self._last_reward_components}
                )

            return observation, float(reward), terminated, truncated, info

        except Exception as e:
            logger.error(f"Error in step(): {str(e)}", exc_info=True)
            # Return a valid but terminal state
            self.done = True
            observation = self._get_observation()
            info = self._get_info()
            info["error"] = str(e)
            return observation, 0.0, True, False, info

    def _update_dbe_state(self) -> None:
        """Update the DBE state with current market conditions."""
        try:
            # Get current market data
            current_prices = self._get_current_prices()
            portfolio_metrics = self.portfolio.get_metrics()

            # Prepare live metrics for DBE
            live_metrics = {
                "step": self.current_step,
                "current_prices": current_prices,
                "portfolio_value": portfolio_metrics.get("total_capital", 0.0),
                "cash": portfolio_metrics.get("cash", 0.0),
                "positions": portfolio_metrics.get("positions", {}),
                "returns": portfolio_metrics.get("returns", 0.0),
                "max_drawdown": portfolio_metrics.get("max_drawdown", 0.0),
            }

            # Add technical indicators if available
            if self.current_data:
                # Get technical indicators from the first available asset
                first_asset = next(iter(self.current_data.keys()))
                if first_asset in self.current_data:
                    first_tf = next(
                        iter(self.current_data[first_asset].keys())
                    )
                    df = self.current_data[first_asset][first_tf]

                    if not df.empty and self.current_step < len(df):
                        current_row = df.iloc[self.current_step]

                        # Add available technical indicators
                        live_metrics.update(
                            {
                                "rsi": current_row.get("rsi", 50.0),
                                "adx": current_row.get("adx", 20.0),
                                "atr": current_row.get("atr", 0.0),
                                "atr_pct": current_row.get("atr_pct", 0.0),
                                "ema_ratio": current_row.get("ema_ratio", 1.0),
                            }
                        )

            # Update DBE state
            self.dbe.update_state(live_metrics)

        except Exception as e:
            logger.warning(f"Failed to update DBE state: {e}")

    def _execute_trades(
        self, action: np.ndarray, dbe_modulation: Dict[str, Any] = None
    ) -> None:
        """Execute trades based on the continuous action vector.

        Args:
            action: Array of continuous values in range [-1, 1] for each asset
                -1 = max sell, 0 = hold, 1 = max buy
            dbe_modulation: Optional dictionary with DBE modulation parameters
        """
        if len(action) != len(self.assets):
            raise ValueError(
                f"Action dimension {len(action)} does not match "
                f"number of assets {len(self.assets)}"
            )

        current_prices = self._get_current_prices()
        portfolio_metrics = self.portfolio.get_metrics()
        cash = portfolio_metrics.get("cash", 0.0)

        # Apply DBE modulation if available
        if dbe_modulation is None:
            dbe_modulation = {}

        # Calculate target positions based on actions
        for _i, (asset, action_value) in enumerate(
            zip(self.assets, action, strict=True)
        ):
            # Clip action to valid range
            action_value = np.clip(float(action_value), -1.0, 1.0)

            # Get current position info
            current_position = portfolio_metrics["positions"].get(
                asset, {"quantity": 0.0})
            current_qty = current_position.get("quantity", 0.0)
            current_value = current_qty * current_prices[asset]

            # Calculate target position value based on action
            # action = -1: Sell all (target_value = 0)
            # action = 0: Keep current position (target_value = current_value)
            # action = 1: Buy with all available cash
            if action_value >= 0:
                # Buy or hold (0 to 1)
                if action_value == 0:
                    # Hold position
                    target_value = current_value
                else:
                    # Buy more based on action value (0 to 1 scale)
                    max_buy_value = cash * action_value
                    target_value = current_value + max_buy_value
            else:
                # Sell (0 to -1)
                sell_pct = abs(action_value)  # Convert to positive percentage
                target_value = current_value * (1 - sell_pct)

            # Convert continuous action to discrete action for OrderManager
            # action_value: -1 to 1 -> discrete_action:
            #   0 (hold), 1 (buy), 2 (sell)
            if action_value > 0.1:  # Buy threshold
                discrete_action = 1
            elif action_value < -0.1:  # Sell threshold
                discrete_action = 2
            else:  # Hold
                discrete_action = 0

            # Execute the trade through the order manager
            self.order_manager.execute_action(
                action=discrete_action,
                current_price=current_prices[asset],
                asset=asset
            )

            # Update cash after each trade (simplified, actual
            # implementation may vary)
            cash -= (target_value - current_value)

        # After all trades, update portfolio metrics
        self.portfolio.update_market_price(current_prices)
        # Rebalance the portfolio after market price update
        self.portfolio.rebalance(current_prices)

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all assets.

        Returns:
            Dictionary mapping asset symbols to their current price.
        """
        prices = {}
        for _asset, timeframe_data in self.current_data.items():
            if not timeframe_data:
                continue

            # Use the first available timeframe
            tf = next(iter(timeframe_data.keys()))
            df = timeframe_data[tf]

            if not df.empty and self.current_step < len(df):
                # Try to get the close price, fall back to first numeric column
                if "close" in df.columns:
                    prices[_asset] = df.iloc[self.current_step]["close"]
                elif not df.empty:
                    # Fall back to first numeric column
                    numeric_cols = df.select_dtypes(
                        include=[np.number]
                    ).columns
                    if len(numeric_cols) > 0:
                        current_row = df.iloc[self.current_step]
                        prices[_asset] = current_row[numeric_cols[0]]

        return prices

    def _get_current_timestamp(self) -> pd.Timestamp:
        """Get the current timestamp.

        Returns:
            The current timestamp from the first available asset/timeframe.

        Raises:
            RuntimeError: If no timestamp data is available.
        """
        for _asset, timeframe_data in self.current_data.items():
            if not timeframe_data:
                continue

            # Use the first available timeframe
            tf = next(iter(timeframe_data.keys()))
            df = timeframe_data[tf]

            if not df.empty and self.current_step < len(df):
                return df.index[self.current_step]

        raise RuntimeError("No timestamp data available")

    def _get_safe_timestamp(self) -> Optional[str]:
        """Get the current timestamp safely.

        Returns:
            str or None: The current timestamp as ISO format string,
                or None if not available.
        """
        try:
            return self._get_current_timestamp().isoformat()
        except Exception:
            return None

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation from the environment by combining the latest
        data from all assets and passing it to the StateBuilder.
        
        The observation is built by:
        1. Getting the required features for each timeframe from the StateBuilder
        2. For each asset and timeframe, selecting the current row of data
        3. Combining all features into a single DataFrame
        4. Passing the combined data to the StateBuilder
        
        Returns:
            np.ndarray: The observation vector
            
        Raises:
            RuntimeError: If the observation cannot be constructed
        """
        cache_key = f"{self.current_chunk_idx}_{self.step_in_chunk}"
        if hasattr(self, '_observation_cache') and cache_key in self._observation_cache:
            return self._observation_cache[cache_key]

        try:
            all_assets_features = []
            
            # Get the feature configuration directly from the StateBuilder
            try:
                feature_config = self.state_builder.get_feature_names()
                if not feature_config:
                    raise ValueError(
                        "Empty feature configuration from StateBuilder"
                    )
                logger.debug(
                    f"Feature config from StateBuilder: {feature_config}"
                )
            except Exception as e:
                logger.error(f"Failed to get feature configuration: {str(e)}")
                raise RuntimeError(f"Could not get feature configuration: {str(e)}")

            for asset in self.assets:
                if asset not in self.current_data:
                    logger.warning(f"No data for asset {asset}, skipping.")
                    continue

                asset_timeframe_data = self.current_data[asset]
                asset_features = {}

                for tf_name, required_features in feature_config.items():
                    if tf_name not in asset_timeframe_data:
                        logger.warning(f"Timeframe {tf_name} not found for asset {asset}.")
                        continue
                    
                    df = asset_timeframe_data[tf_name]
                    if df.empty or self.step_in_chunk >= len(df):
                        logger.warning(f"No data available for {asset} {tf_name} at step {self.step_in_chunk}")
                        continue

                    # Get the current row of data
                    current_row = df.iloc[self.step_in_chunk]
                    
                    # Check for missing features but don't fail if some are missing
                    missing = [f for f in required_features if f not in current_row.index]
                    if missing:
                        logger.debug(f"Missing features for {asset} in {tf_name}: {missing}")
                    
                    # Only include features that exist in the data
                    available_features = [f for f in required_features if f in current_row.index]
                    if not available_features:
                        logger.warning(f"No available features for {asset} in {tf_name}")
                        continue
                    
                    # Select the available features
                    selected_features = current_row[available_features]
                    
                    # Add to asset features with timeframe prefix
                    for feature, value in selected_features.items():
                        asset_features[f"{tf_name}_{feature}"] = value
                
                # Add asset prefix and append to all assets
                if asset_features:
                    asset_df = pd.Series(asset_features, name=asset).to_frame().T
                    all_assets_features.append(asset_df)
                else:
                    logger.warning(f"No features available for asset {asset}")

            if not all_assets_features:
                logger.error("No market data available to build observation.")
                # Return an array of NaNs with the correct shape if no data is available
                return np.full(self.observation_space.shape, np.nan, dtype=np.float32)

            # Combine all assets into a single DataFrame
            combined_data = pd.concat(all_assets_features, axis=0)
            
            # Check if combined_data is empty after concatenation
            if combined_data.empty:
                logger.error("Combined data is empty after processing. Cannot build observation.")
                return np.full(self.observation_space.shape, np.nan, dtype=np.float32)
            
            # Log the shape and columns of the combined data for debugging
            logger.debug(f"Combined data shape: {combined_data.shape}")
            logger.debug(f"Combined data columns: {combined_data.columns.tolist()}")
            
            # Build the observation using the StateBuilder
            observation = self.state_builder.build_observation(combined_data)
            
            # Cache the observation
            self._observation_cache[cache_key] = observation
            
            # Log the observation shape for debugging
            logger.debug(f"Observation shape: {observation.shape}")
            
            # === DEBUG SHAPE LOGGING ===
            expected_shape = self.observation_space.shape
            logger.info(f"[Env] Shape attendue (flatten): {expected_shape}")
            # Après appel à StateBuilder
            logger.info(f"[Env] Shape 1D retournée par StateBuilder: {observation.shape}, taille: {observation.size}")
            assert observation.shape == expected_shape, (
                f"[Env] Mismatch de dimension: {observation.shape} vs attendu {expected_shape}")
            
            return observation

        except Exception as e:
            logger.error(f"Error in _get_observation: {str(e)}", exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate the reward for the current step.

        The reward is composed of several components:
        1. Portfolio returns (scaled by return_scale)
        2. Risk penalty (based on max drawdown)
        3. Transaction cost penalty
        4. Position concentration penalty
        5. Action smoothness penalty

        Args:
            action: Array of continuous values in range [-1, 1] for each asset
                -1 = max sell, 0 = hold, 1 = max buy

        Returns:
            float: The calculated reward for the current step
        """
        # Get portfolio metrics
        portfolio_metrics = self.portfolio.get_metrics()

        # 1. Base reward from portfolio returns
        returns = portfolio_metrics.get("returns", 0.0)
        max_drawdown = portfolio_metrics.get("max_drawdown", 0.0)

        # 2. Calculate risk-adjusted returns using the reward calculator
        reward = self.reward_calculator.calculate(
            returns=returns,
            risk_free_rate=self.config.get("environment", {}).get(
                "risk_free_rate", 0.0),
            max_drawdown=max_drawdown,
        )

        # 3. Transaction cost penalty (encourage fewer, larger trades)
        transaction_cost_penalty = 0.0
        has_history = hasattr(self.portfolio, 'transaction_history')
        if has_history and self.portfolio.transaction_history:
            transaction_cost_penalty = (
                -0.01 * len(self.portfolio.transaction_history)
                / max(1, self.current_step)
            )

        # 4. Position concentration penalty (encourage diversification)
        position_concentration_penalty = 0.0
        if portfolio_metrics.get("positions"):
            position_values = [
                pos.get("market_value", 0)
                for pos in portfolio_metrics["positions"].values()
            ]
            if position_values:
                # Calculate Herfindahl-Hirschman Index (HHI) for positions
                total_value = sum(position_values)
                if total_value > 0:
                    hhi = sum((v / total_value) ** 2 for v in position_values)
                    # Normalize to [0, 1] where 1 is perfect concentration,
                    # 1/n is perfect diversification
                    n = len(position_values)
                    hhi_normalized = (
                        (hhi - 1.0 / n) / (1.0 - 1.0 / n)
                        if n > 1 else 1.0
                    )
                    position_concentration_penalty = -0.1 * hhi_normalized

        # 5. Action smoothness penalty (encourage less frequent
        # large position changes)
        action_smoothness_penalty = 0.0
        if hasattr(self, '_last_action') and self._last_action is not None:
            # Penalize large changes in action values
            action_diff = np.mean(
                np.abs(np.array(action) - np.array(self._last_action))
            )
            action_smoothness_penalty = -0.05 * action_diff
        self._last_action = action.copy()

        # Store reward components in info for debugging
        reward_components = {
            'base_reward': reward,
            'transaction_cost_penalty': transaction_cost_penalty,
            'position_concentration_penalty': position_concentration_penalty,
            'action_smoothness_penalty': action_smoothness_penalty
        }

        # Get weights from config or use defaults
        reward_weights = self.config.get("environment", {}).get(
            "reward_weights", {
                'base_weight': 1.0,
                'transaction_cost_weight': 1.0,
                'concentration_weight': 0.5,
                'smoothness_weight': 0.2
            })

        # Calculate weighted reward
        total_reward = (
            reward_weights.get('base_weight', 1.0) * reward
            + reward_weights.get('transaction_cost_weight', 1.0)
            * transaction_cost_penalty
            + reward_weights.get('concentration_weight', 0.5)
            * position_concentration_penalty
            + reward_weights.get('smoothness_weight', 0.2)
            * action_smoothness_penalty
        )

        # Store reward components in info for debugging
        self._last_reward_components = reward_components

        return float(total_reward)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state.

        Returns:
            Dictionary containing various metrics and state information useful
            for debugging and monitoring the environment.
        """
        portfolio_metrics = self.portfolio.get_metrics()
        current_prices = self._get_current_prices()

        # Calculate position values
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
                    )
                }
                total_position_value += value

        # Get reward components if available
        reward_components = {}
        if hasattr(self, '_last_reward_components'):
            reward_components = self._last_reward_components

        # Get action stats if available
        action_stats = {}
        if hasattr(self, '_last_action') and self._last_action is not None:
            action = self._last_action
            action_stats = {
                "action_mean": float(np.mean(action)),
                "action_std": float(np.std(action)),
                "action_min": float(np.min(action)),
                "action_max": float(np.max(action)),
                "num_assets": len(action)
            }

        # Compile all info
        info = {
            # Environment state
            "step": self.current_step,
            "chunk": self.current_chunk,
            "done": self.done,

            # Portfolio metrics
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

            # Position information
            "positions": position_values,

            # Current market state
            "market": {
                "num_assets": len(current_prices),
                "assets": list(current_prices.keys()),
                "current_prices": current_prices,
            },

            # Action information
            "action_stats": action_stats,

            # Reward components
            "reward_components": reward_components,

            # Performance metrics
            "performance": {
                "timestamp": self._get_safe_timestamp(),
                "steps_per_second": (self.current_step / max(
                    0.0001, time.time() - self._episode_start_time
                ) if hasattr(self, '_episode_start_time') else 0.0),
            }
        }

        return info

    def render(self, mode: str = "human") -> None:
        """Render the environment."""
        if mode == "human":
            portfolio_value = self.portfolio.get_portfolio_value()
            print(
                f"Step: {self.current_step}, "
                f"Portfolio Value: {portfolio_value:.2f}, "
                f"Cash: {self.portfolio.cash:.2f}, "
                f"Positions: {self.portfolio.positions}"
            )

    def close(self) -> None:
        """Clean up resources."""
        pass  # No special cleanup needed

