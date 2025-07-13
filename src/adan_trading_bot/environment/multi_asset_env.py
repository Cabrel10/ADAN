#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Gymnasium environment for the ADAN trading bot.

This file defines the core trading environment, which orchestrates data loading,
feature engineering, state representation, action handling, and reward calculation.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from ..data_processing.chunked_loader import ChunkedDataLoader
from .state_builder import StateBuilder
from ..environment.reward_calculator import RewardCalculator
from ..portfolio.portfolio_manager import PortfolioManager
from ..trading.order_manager import OrderManager

logger = logging.getLogger(__name__)

class AdanTradingEnv(gym.Env):
    """
    A comprehensive, multi-asset trading environment for reinforcement learning.

    This environment integrates all the components of the trading system:
    - Data Loading: Uses ChunkedDataLoader to get data in chunks for memory efficiency.
    - State Management: Uses StateBuilder to construct 3D observation space.
    - Portfolio Management: Uses PortfolioManager to track finances across multiple assets.
    - Order Execution: Uses OrderManager to simulate trades with fees and slippage.
    - Reward Calculation: Uses RewardCalculator for reward shaping with optimal trade bonus.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Dict[str, Any], mode: str = 'train'):
        """
        Initialize the environment.

        Args:
            config: Configuration dictionary with sections for data, portfolio, trading, etc.
            mode: Either 'train', 'val', or 'test' to determine which data split to use.
        """
        super().__init__()
        self.config = config
        self.mode = mode
        
        # Initialize components
        self._init_components()
        
        # Initialize state tracking
        self.current_step = 0
        self.step_in_chunk = 0
        self.current_chunk_idx = 0
        self.optimal_chunk_pnl = 0.0
        self.done = False
        
        logger.info(f"AdanTradingEnv initialized in {mode} mode")
    
    def _init_components(self) -> None:
        """Initialize all environment components."""
        # 1. Initialize data loader for the current split
        self._init_data_loader()
        
        # 2. Initialize state builder with window size
        self.state_builder = StateBuilder(
            window_size=self.config['state'].get('window_size', 30),
            timeframes=self.config['data'].get('timeframes', ['5m', '1h', '4h'])
        )
        
        # 3. Initialize portfolio manager
        self.portfolio = PortfolioManager(
            initial_balance=self.config['portfolio'].get('initial_balance', 10000.0),
            max_leverage=self.config['portfolio'].get('max_leverage', 3.0),
            risk_per_trade=self.config['portfolio'].get('risk_per_trade', 0.01)
        )
        
        # 4. Initialize order manager
        self.order_manager = OrderManager(
            slippage=self.config['trading'].get('slippage', 0.0005),
            commission=self.config['trading'].get('commission', 0.0005)
        )
        
        # 5. Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            config=self.config.get('rewards', {})
        )
        
        # Set up action and observation spaces
        self._setup_spaces()
    
    def _init_data_loader(self) -> None:
        """
        Initialize the chunked data loader for the current split.
        
        This method sets up the ChunkedDataLoader with the appropriate configuration
        for the current mode (train/val/test) and extracts the list of available assets.
        """
        data_config = self.config['data']
        
        # Get the base data directory
        base_data_dir = Path(data_config.get('data_dir', 'data/final'))
        
        # Get features configuration
        features_config = self._get_features_config()
        
        # Initialize the data loader with the new structure
        self.data_loader = ChunkedDataLoader(
            data_dir=base_data_dir,
            chunk_size=data_config.get('chunk_size', 10000),
            assets_list=data_config.get('assets'),
            features_by_timeframe=features_config,
            split=self.mode,  # 'train', 'val', or 'test'
            timeframes=data_config.get('timeframes', ['5m', '1h', '4h'])
        )
        
        # Store assets list for easy access
        self.assets = self.data_loader.assets_list
        if not self.assets:
            raise ValueError("No assets available for trading")
            
        logger.info(f"Initialized {self.mode} data loader with {len(self.assets)} assets")
        logger.debug(f"Available assets: {self.assets}")
    
    def _get_features_config(self) -> Dict[str, List[str]]:
        """Extract features configuration from the main config."""
        features_config = {}
        
        if 'feature_engineering' in self.config and 'timeframes' in self.config['feature_engineering']:
            for tf in self.config['feature_engineering']['timeframes']:
                features_config[tf] = self.config['feature_engineering'].get('features', {}).get(tf, [
                    'open', 'high', 'low', 'close', 'volume'
                ])
        else:
            # Default features if not specified
            features_config = {
                '5m': ['open', 'high', 'low', 'close', 'volume'],
                '1h': ['open', 'high', 'low', 'close'],
                '4h': ['open', 'close']
            }
            
        return features_config
    
    def _setup_spaces(self) -> None:
        """Set up action and observation spaces."""
        # Action space: [hold, buy, sell] for each asset
        self.action_space = spaces.MultiDiscrete(
            [3] * len(self.assets)  # 0=hold, 1=buy, 2=sell for each asset
        )
        
        # Get observation space shape from state builder
        obs_shape = self.state_builder.get_observation_shape()
        
        # Create observation space as a dictionary with an entry per asset
        self.observation_space = spaces.Dict({
            asset: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            )
            for asset in self.assets
        })
    
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        This method performs the following steps:
        1. Resets all environment state variables
        2. Resets the portfolio
        3. Loads the first chunk of data
        4. Initializes chunk metrics
        5. Returns the initial observation and info
        
        Args:
            **kwargs: Additional arguments that might be used by wrappers
            
        Returns:
            Tuple containing:
            - observation: Initial observation for each asset
            - info: Additional information about the environment state
            
        Raises:
            RuntimeError: If the environment cannot be reset (e.g., no data available)
        """
        try:
            # Reset environment state
            self.current_step = 0
            self.step_in_chunk = 0
            self.current_chunk_idx = 0
            self.done = False
            
            # Reset reward history
            if hasattr(self, 'reward_history'):
                self.reward_history = []
            
            # Reset portfolio
            if not hasattr(self, 'portfolio'):
                raise RuntimeError("Portfolio not initialized. Call _init_components() first.")
                
            self.portfolio.reset()
            
            # Reset data loader if needed
            if not hasattr(self, 'data_loader'):
                self._init_data_loader()
            
            # Load the first chunk of data
            self.current_chunk = self.data_loader.load_chunk(0)
            if not self.current_chunk:
                raise ValueError("No data available in the data loader")
            
            # Log chunk information
            chunk_start = next(iter(self.current_chunk.values()))['timestamp'].iloc[0]
            chunk_end = next(iter(self.current_chunk.values()))['timestamp'].iloc[-1]
            logger.info(
                f"Reset environment. Loaded chunk 0 with data from {chunk_start} to {chunk_end} "
                f"({len(next(iter(self.current_chunk.values())))} steps)"
            )
            
            # Initialize chunk metrics
            self._initialize_chunk_metrics()
            
            # Initialize observation space if not already done
            if not hasattr(self, 'observation_space'):
                self._initialize_observation_space()
            
            # Get initial observation
            observation = self._get_observation()
            info = self._get_info()
            
            # Log successful reset
            logger.info("Environment reset completed successfully")
            
            return observation, info
            
        except Exception as e:
            error_msg = f"Error resetting environment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Action to take (0=hold, 1=buy, 2=sell for each asset)
            
        Returns:
            observation: Next observation for each asset
            reward: Reward for the current step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode has already terminated. Call reset() to start a new episode.")
        
        # Execute trades based on actions
        self._execute_trades(action)
        
        # Move to next step
        self.current_step += 1
        self.step_in_chunk += 1
        
        # Check if we need to load the next chunk
        chunk_size = len(next(iter(self.current_chunk.values())))
        if self.step_in_chunk >= chunk_size:
            # Calculate agent's PnL for the completed chunk
            agent_chunk_pnl = self.portfolio.get_portfolio_value() - self.portfolio.initial_balance
            
            # Calculate optimal trade bonus (if any)
            optimal_bonus = self.reward_calculator.calculate_optimal_bonus(
                agent_pnl=agent_chunk_pnl,
                optimal_pnl=self.optimal_chunk_pnl
            )
            
            # Store for info
            self.last_chunk_pnl = agent_chunk_pnl
            self.last_optimal_bonus = optimal_bonus
            
            # Load next chunk
            self.current_chunk_idx += 1
            try:
                self.current_chunk = self.data_loader.load_chunk(self.current_chunk_idx)
                self.step_in_chunk = 0
                
                # Calculate optimal PnL for the new chunk
                self.optimal_chunk_pnl = self.data_loader.calculate_optimal_pnl_for_chunk(
                    self.current_chunk
                )
            except IndexError:
                # No more chunks available
                self.done = True
        
        # Calculate reward for this step (includes any optimal trade bonus)
        reward = self._calculate_reward()
        
        # Get next observation
        observation = self._get_observation()
        
        # Check if episode is done
        terminated = self.done
        truncated = False  # Can be set by wrappers
        
        # Get additional info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _init_components(self) -> None:
        """Initialize all environment components."""
        # 1. Initialize data loader for the current split
        self._init_data_loader()
        
        # 2. Initialize state builder with window size
        self.state_builder = StateBuilder(
            window_size=self.config['state'].get('window_size', 30),
            timeframes=self.config['data'].get('timeframes', ['5m', '1h', '4h'])
        )
        
        # 3. Initialize portfolio manager
        self.portfolio = PortfolioManager(
            initial_balance=self.config['portfolio'].get('initial_balance', 10000.0),
            max_leverage=self.config['portfolio'].get('max_leverage', 3.0),
            risk_per_trade=self.config['portfolio'].get('risk_per_trade', 0.01)
        )
        
        # 4. Initialize order manager
        self.order_manager = OrderManager(
            slippage=self.config['trading'].get('slippage', 0.0005),
            commission=self.config['trading'].get('commission', 0.0005)
        )
        
        # 5. Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            config=self.config.get('rewards', {})
        )
        
        # 5. Initialize state builder
        self.state_builder = StateBuilder(
            config=self.config.get('state', {})
        )
    
    def _init_data_loader(self) -> None:
        """Initialize the chunked data loader."""
        data_config = self.config['data']
        
        # Get timeframes from config or use default
        timeframes = self.config.get('feature_engineering', {}).get('timeframes', ['5m', '1h', '4h'])
        
        # Initialize the data loader with correct parameters
        self.data_loader = ChunkedDataLoader(
            data_dir=Path(data_config.get('data_dir', 'data/final')),
            chunk_size=data_config.get('chunk_size', 10000),
            assets_list=data_config.get('assets'),
            features_by_timeframe=self._get_features_config(),
            split=self.mode,  # 'train', 'val', or 'test'
            timeframes=timeframes
        )
        
        # Store assets list for easy access
        self.assets = self.data_loader.assets_list
        if not self.assets:
            raise ValueError("No assets available for trading")
            
        logger.info(f"Initialized data loader with {len(self.assets)} assets and timeframes: {timeframes}")
    
    def _get_features_config(self) -> Dict[str, List[str]]:
        """Extract features configuration from the main config."""
        features_config = {}
        
        if 'feature_engineering' in self.config and 'timeframes' in self.config['feature_engineering']:
            for tf in self.config['feature_engineering']['timeframes']:
                features_config[tf] = self.config['feature_engineering'].get('features', {}).get(tf, [
                    'open', 'high', 'low', 'close', 'volume'
                ])
        else:
            # Default features if not specified
            features_config = {
                '5m': ['open', 'high', 'low', 'close', 'volume'],
                '1h': ['open', 'high', 'low', 'close'],
                '4h': ['open', 'close']
            }
            
        return features_config
    
    def _setup_spaces(self) -> None:
        """Set up action and observation spaces."""
        # Action space: [hold, buy, sell] for each asset
        self.action_space = spaces.MultiDiscrete(
            [3] * len(self.assets)  # 0=hold, 1=buy, 2=sell for each asset
        )
        
        # Get observation space shape from state builder
        obs_shape = self.state_builder.get_observation_shape()
        
        # Create observation space as a dictionary with an entry per asset
        self.observation_space = spaces.Dict({
            asset: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            )
            for asset in self.assets
        })
        
        # Load the first chunk of data
        self.current_data = self.data_loader.load_chunk(0)
        
        # Initialize observation space
        self._initialize_observation_space()
    
    def _initialize_observation_space(self) -> None:
        """
        Initialize the observation space based on the first data point.
        
        This method sets up the observation space as a dictionary of Box spaces,
        one for each asset, with shapes determined by the StateBuilder.
        """
        if not self.current_data:
            raise ValueError("No data available to initialize observation space")
        
        # Get state shape from the state builder
        sample_asset = next(iter(self.current_data.keys()))
        sample_data = self.current_data[sample_asset].iloc[0:1]
        state_shape = self.state_builder.get_observation_shape()
        
        # Create observation space as a dictionary with an entry per asset
        self.observation_space = spaces.Dict({
            asset: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=state_shape,
                dtype=np.float32
            )
            for asset in self.assets
        })

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        This method executes the following steps:
        1. Validates the current environment state
        2. Executes trades based on the action vector
        3. Updates the environment state
        4. Calculates the reward
        5. Checks termination conditions
        
        Args:
            action: Array of actions (0=hold, 1=buy, 2=sell) for each asset
            
        Returns:
            Tuple containing:
            - observation: Dict mapping asset names to their 3D observation tensors
            - reward: Float representing the reward for the current step
            - terminated: Boolean indicating if the episode has ended
            - truncated: Boolean indicating if the episode was truncated
            - info: Dictionary containing diagnostic information
            
        Raises:
            RuntimeError: If the environment is in an invalid state
            ValueError: If the action is invalid
        """
        if not hasattr(self, 'current_data') or not self.current_data:
            raise RuntimeError("Environment not properly initialized. Call reset() first.")
            
        if self.done:
            raise RuntimeError("Episode has already terminated. Call reset() to start a new episode.")
        
        # Validate action
        if not isinstance(action, (np.ndarray, list)):
            raise ValueError(f"Action must be a numpy array or list, got {type(action)}")
            
        if len(action) != len(self.assets):
            raise ValueError(
                f"Action dimension {len(action)} does not match "
                f"number of assets {len(self.assets)}"
            )
        
        try:
            # Execute trades based on actions
            self._execute_trades(np.asarray(action))
            
            # Update step counters
            self.current_step += 1
            self.step_in_chunk += 1
            
            # Check if we need to load the next chunk
            current_chunk_length = len(next(iter(self.current_data.values())))
            if self.step_in_chunk >= current_chunk_length:
                self._handle_chunk_transition()
            
            # Get next observation
            observation = self._get_observation()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check if episode is done
            terminated = self.done
            truncated = False  # Can be set by wrappers or custom logic
            
            # Get additional info
            info = self._get_info()
            
            # Add step information to info
            info.update({
                'current_step': self.current_step,
                'step_in_chunk': self.step_in_chunk,
                'chunk_idx': self.current_chunk_idx,
                'timestamp': self._get_current_timestamp().isoformat()
            })
            
            # Log progress
            if self.current_step % 100 == 0:
                logger.info(
                    f"Step {self.current_step} (Chunk {self.current_chunk_idx}, "
                    f"Step {self.step_in_chunk}): Reward={reward:.4f}, "
                    f"Portfolio Value={self.portfolio.portfolio_value:.2f}"
                )
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in step {self.current_step}: {str(e)}")
            # Return a zero observation and negative reward on error
            obs_shape = self.state_builder.get_observation_shape()
            zero_obs = {asset: np.zeros(obs_shape, dtype=np.float32) for asset in self.assets}
            return zero_obs, -10.0, True, False, {'error': str(e)}

    def _initialize_chunk_metrics(self) -> None:
        """
        Initialize metrics for the current chunk.
        
        This method is called at the start of each new chunk to set up tracking
        of chunk-specific metrics like optimal PnL and bonus calculations.
        It performs the following operations:
        1. Calculates the optimal PnL for the chunk based on price movements
        2. Initializes tracking metrics for the chunk
        3. Sets up reward calculation parameters
        4. Logs the initialization details
        
        The optimal PnL is calculated based on the highest frequency timeframe
        available in the data (typically 5m). This provides a realistic benchmark
        for the agent's performance.
        """
        if not hasattr(self, 'portfolio') or not hasattr(self, 'current_data') or not self.current_data:
            logger.warning("Cannot initialize chunk metrics: Portfolio or data not available")
            return
            
        try:
            # Get the highest frequency timeframe (first in the list)
            highest_tf = self.state_builder.timeframes[0] if hasattr(self.state_builder, 'timeframes') else '5m'
            close_col = f"{highest_tf}_close"
            
            # Get the first available asset's data for optimal PnL calculation
            asset_name, chunk_data = next(iter(self.current_data.items()))
            
            if chunk_data.empty:
                logger.warning(f"Empty data for {asset_name} in chunk {self.current_chunk_idx}")
                return
                
            # Ensure we have the required columns
            if close_col not in chunk_data.columns:
                # Fall back to any close column if the exact one isn't found
                close_cols = [col for col in chunk_data.columns if col.endswith('_close')]
                if not close_cols:
                    logger.warning(f"No close price column found in chunk {self.current_chunk_idx}")
                    return
                close_col = close_cols[0]
            
            # Calculate price changes and optimal PnL
            prices = chunk_data[close_col]
            if len(prices) < 2:
                logger.warning(f"Not enough data points for PnL calculation in chunk {self.current_chunk_idx}")
                return
                
            price_changes = prices.pct_change().dropna()
            
            if len(price_changes) == 0:
                logger.warning(f"No valid price changes in chunk {self.current_chunk_idx}")
                return
            
            # Calculate optimal PnL: sum of all positive returns (perfect trades)
            optimal_returns = (1 + price_changes[price_changes > 0]).prod()
            optimal_pnl = (optimal_returns - 1) * self.portfolio.initial_value
            optimal_pnl = max(0, optimal_pnl)  # Ensure non-negative
            
            # Get reward configuration with defaults
            reward_config = self.config.get('reward', {})
            max_chunk_bonus = reward_config.get('max_chunk_bonus', 10.0)
            min_chunk_size = reward_config.get('min_chunk_size', 100)  # Minimum steps for meaningful PnL
            
            # Adjust bonus based on chunk size
            chunk_size = len(chunk_data)
            size_factor = min(1.0, chunk_size / min_chunk_size) if min_chunk_size > 0 else 1.0
            adjusted_bonus = max_chunk_bonus * size_factor
            
            # Set chunk metrics
            self.chunk_metrics = {
                'start_value': float(self.portfolio.get_portfolio_value()),
                'start_cash': float(self.portfolio.cash),
                'optimal_pnl': float(optimal_pnl),
                'max_bonus': float(adjusted_bonus),
                'start_step': int(self.current_step),
                'start_timestamp': self._get_current_timestamp().isoformat(),
                'chunk_size': int(chunk_size),
                'asset': str(asset_name),
                'price_range': f"{prices.min():.8f}-{prices.max():.8f}",
                'volatility': float(price_changes.std() * np.sqrt(365 * 24 * 60 / 5))  # Annualized
            }
            
            # Log detailed initialization info
            logger.info(
                f"Initialized chunk {self.current_chunk_idx} metrics:\n"
                f"  Asset: {asset_name} ({chunk_size} steps, {highest_tf} timeframe)\n"
                f"  Price range: {prices.min():.8f} - {prices.max():.8f} "
                f"({((prices.max() - prices.min()) / prices.min() * 100):.2f}%)\n"
                f"  Volatility: {self.chunk_metrics['volatility']:.2%} (annualized)\n"
                f"  Start value: {self.chunk_metrics['start_value']:.2f} "
                f"(Cash: {self.chunk_metrics['start_cash']:.2f})\n"
                f"  Target PnL: {optimal_pnl:.2f} (max bonus: {adjusted_bonus:.2f})"
            )
            
            # Log sample price changes for debugging
            if logger.isEnabledFor(logging.DEBUG):
                sample_changes = price_changes.head(5).to_list()
                logger.debug(
                    f"Sample price changes for {asset_name} (first 5): "
                    f"{[f'{x:.2%}' for x in sample_changes]}"
                )
                
        except Exception as e:
            logger.error(
                f"Error initializing metrics for chunk {self.current_chunk_idx}: {str(e)}",
                exc_info=True
            )
            # Initialize with safe defaults
            self.chunk_metrics = {
                'start_value': float(self.portfolio.get_portfolio_value()),
                'start_cash': float(self.portfolio.cash),
                'optimal_pnl': 0.0,
                'max_bonus': 0.0,
                'start_step': int(self.current_step),
                'start_timestamp': self._get_current_timestamp().isoformat(),
                'error': str(e)
            }
    
    def _handle_chunk_transition(self) -> None:
        """
        Handle the transition to the next chunk of data.
        
        This method is called when we've reached the end of the current chunk
        and need to load the next one or terminate the episode. It performs the
        following steps:
        1. Saves metrics for the completed chunk
        2. Loads the next chunk of data
        3. Validates the loaded data
        4. Initializes metrics for the new chunk
        5. Updates portfolio and other components
        
        If any error occurs during the transition, the episode is marked as done.
        """
        # Save chunk metrics before transitioning
        prev_chunk_idx = self.current_chunk_idx
        
        try:
            if hasattr(self, 'chunk_metrics'):
                chunk_pnl = self.portfolio.get_portfolio_value() - self.chunk_metrics['start_value']
                chunk_duration = self.current_step - self.chunk_metrics.get('start_step', 0)
                optimal_pnl = self.chunk_metrics.get('optimal_pnl', 0)
                pnl_ratio = (chunk_pnl / optimal_pnl) * 100 if optimal_pnl > 0 else 0.0
                
                logger.info(
                    f"Completed chunk {prev_chunk_idx} (Steps: {chunk_duration}):\n"
                    f"  PnL: {chunk_pnl:+.2f} ({pnl_ratio:.1f}% of optimal {optimal_pnl:.2f})\n"
                    f"  Portfolio: {self.portfolio.get_portfolio_value():.2f} "
                    f"(Start: {self.chunk_metrics['start_value']:.2f}, "
                    f"Change: {chunk_pnl/self.chunk_metrics['start_value']*100 if self.chunk_metrics['start_value'] > 0 else 0:.1f}%)\n"
                    f"  Positions: {len(self.portfolio.positions)} active"
                )
                
                # Log detailed position information
                if self.portfolio.positions:
                    pos_info = []
                    for asset, pos in self.portfolio.positions.items():
                        if pos.amount > 0:  # Only show open positions
                            pnl_pct = (pos.current_value / pos.cost_basis - 1) * 100 if pos.cost_basis > 0 else 0
                            pos_info.append(
                                f"{asset}: {pos.amount:.4f} @ {pos.avg_price:.8f} "
                                f"(Cur: {pos.current_price:.8f}, PnL: {pnl_pct:+.1f}%)"
                            )
                    if pos_info:
                        logger.info("  Positions detail:\n  - " + "\n  - ".join(pos_info))
            
            # Move to next chunk
            self.current_chunk_idx += 1
            
            # Check if we've reached the end of available chunks
            if self.current_chunk_idx >= len(self.data_loader):
                self.done = True
                logger.info(
                    f"Episode complete: Reached the end of available data "
                    f"(chunk {prev_chunk_idx} of {len(self.data_loader)-1})"
                )
                return
                
            # Load the next chunk
            logger.debug(f"Loading chunk {self.current_chunk_idx}...")
            self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)
            self.step_in_chunk = 0
            
            # Validate the loaded chunk
            if not self.current_data:
                raise ValueError(f"Chunk {self.current_chunk_idx} is empty")
                
            # Log chunk transition details
            first_asset = next(iter(self.current_data.keys()))
            chunk_df = self.current_data[first_asset]
            chunk_start = chunk_df['timestamp'].iloc[0] if 'timestamp' in chunk_df.columns else 'N/A'
            chunk_end = chunk_df['timestamp'].iloc[-1] if 'timestamp' in chunk_df.columns else 'N/A'
            num_steps = len(chunk_df)
            
            # Log available assets and their data points
            asset_info = []
            for asset, df in self.current_data.items():
                asset_info.append(f"{asset}: {len(df)} steps")
                
                # Log sample data for the first few assets
                if len(asset_info) <= 3 and not df.empty:
                    logger.debug(
                        f"Sample data for {asset} (chunk {self.current_chunk_idx}):\n"
                        f"  Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}\n"
                        f"  Columns: {', '.join(df.columns)}\n"
                        f"  First row: {df.iloc[0].to_dict()}\n"
                        f"  Last row: {df.iloc[-1].to_dict()}"
                    )
            
            logger.info(
                f"Loaded chunk {self.current_chunk_idx} with {len(self.current_data)} assets:\n"
                f"  Time range: {chunk_start} to {chunk_end} ({num_steps} steps)\n"
                f"  Assets: {', '.join(asset_info[:5])}"
                + (f" (+{len(asset_info)-5} more)" if len(asset_info) > 5 else "")
            )
            
            # Initialize metrics for the new chunk
            self._initialize_chunk_metrics()
            
            # Update components with new chunk information
            if hasattr(self.portfolio, 'on_new_chunk'):
                self.portfolio.on_new_chunk(self.current_chunk_idx)
                
            # Log memory usage for debugging
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                logger.debug(
                    f"Memory usage after loading chunk {self.current_chunk_idx}: "
                    f"RSS={mem_info.rss / 1024 / 1024:.1f}MB, "
                    f"VMS={mem_info.vms / 1024 / 1024:.1f}MB"
                )
            except ImportError:
                pass  # psutil not available, skip memory logging
                
        except Exception as e:
            logger.error(
                f"Error during transition to chunk {self.current_chunk_idx}: {str(e)}\n"
                f"Current chunk: {prev_chunk_idx}, Step: {self.current_step}",
                exc_info=True
            )
            self.done = True
            raise  # Re-raise to be handled by the caller
    
    def _execute_trades(self, action: np.ndarray) -> None:
        """
        Execute trades based on the action vector.
        
        This method processes each action in the action vector and executes the corresponding
        trade (buy/sell) for each asset. It handles errors gracefully and logs all actions.
        
        Args:
            action: Array of actions (0=hold, 1=buy, 2=sell) for each asset
            
        Raises:
            ValueError: If the action vector has incorrect dimensions
            RuntimeError: If there's an error executing a trade
        """
        if not hasattr(self, 'assets') or not self.assets:
            raise RuntimeError("No assets available for trading")
            
        if len(action) != len(self.assets):
            raise ValueError(
                f"Action dimension {len(action)} does not match number of assets {len(self.assets)}"
            )
        
        try:
            # Get current prices and timestamp once for all assets
            current_prices = self._get_current_prices()
            current_timestamp = self._get_current_timestamp()
            
            # Process each asset's action
            for asset, action_idx in zip(self.assets, action):
                try:
                    if action_idx == 0:  # Hold
                        continue
                        
                    if asset not in current_prices:
                        logger.warning(f"No price available for asset {asset}, skipping trade")
                        continue
                        
                    price = current_prices[asset]
                    
                    # Log the action for debugging
                    action_str = {1: 'BUY', 2: 'SELL'}.get(action_idx, 'INVALID')
                    logger.debug(
                        f"Executing {action_str} order for {asset} at price {price:.8f} "
                        f"(Step {self.current_step}, Chunk {self.current_chunk_idx})"
                    )
                    
                    if action_idx == 1:  # Buy
                        position_size = self.portfolio.calculate_position_size(price)
                        if position_size > 0:
                            self.order_manager.enter_trade(
                                asset=asset,
                                price=price,
                                size=position_size,
                                direction='long',
                                timestamp=current_timestamp
                            )
                            logger.info(
                                f"Entered long position: {asset} x{position_size:.6f} @ {price:.8f} "
                                f"(Value: {position_size * price:.2f})"
                            )
                        
                    elif action_idx == 2:  # Sell
                        # Check if we have a position to close
                        if asset in self.portfolio.positions and self.portfolio.positions[asset].amount > 0:
                            self.order_manager.exit_trade(
                                asset=asset,
                                price=price,
                                timestamp=current_timestamp
                            )
                            logger.info(f"Exited position: {asset} @ {price:.8f}")
                        else:
                            logger.debug(f"No position to close for {asset}, ignoring SELL action")
                    
                    else:
                        logger.warning(f"Invalid action {action_idx} for asset {asset}")
                        
                except Exception as e:
                    logger.error(
                        f"Error executing trade for {asset} (action={action_idx}): {str(e)}",
                        exc_info=True
                    )
                    # Continue with other assets even if one fails
                    
        except Exception as e:
            logger.error(f"Error in _execute_trades: {str(e)}", exc_info=True)
            # Re-raise to be handled by the caller
            raise

    def _calculate_reward(self) -> float:
        """
        Calculate the reward for the current step with multi-timeframe support.
        
        This enhanced method computes rewards based on:
        1. Multi-timeframe portfolio performance metrics
        2. Risk-adjusted returns across different time horizons
        3. Drawdown analysis per timeframe
        4. Signal consistency across timeframes
        5. Position sizing and risk management
        
        The reward calculation considers:
        - Short-term (5m), medium-term (1h), and long-term (4h) performance
        - Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
        - Drawdown analysis and recovery factors
        - Position concentration and diversification
        - Transaction costs and slippage impact
        
        Returns:
            float: The calculated reward, incorporating multi-timeframe analysis
            
        Raises:
            RuntimeError: If required components are not properly initialized
        """
        if not hasattr(self, 'portfolio') or not hasattr(self, 'reward_calculator'):
            raise RuntimeError("Portfolio or reward calculator not initialized")
            
        try:
            # Initialize metrics dictionary
            metrics = {}
            
            # 1. Basic Portfolio Metrics
            portfolio_value = self.portfolio.get_portfolio_value()
            current_pnl = portfolio_value - self.portfolio.initial_value
            returns = self.portfolio.returns if hasattr(self.portfolio, 'returns') else []
            
            # 2. Multi-timeframe Analysis
            timeframes = getattr(self.state_builder, 'timeframes', ['5m', '1h', '4h'])
            timeframe_metrics = {}
            
            for tf in timeframes:
                tf_returns = getattr(self.portfolio, f'returns_{tf}', returns)
                tf_vol = np.std(tf_returns) * np.sqrt(252 * 24 * (60 / int(tf[:-1]))) if tf_returns else 0
                
                timeframe_metrics[tf] = {
                    'returns': tf_returns[-1] if tf_returns else 0,
                    'volatility': tf_vol,
                    'sharpe': (np.mean(tf_returns) / (tf_vol + 1e-9) * np.sqrt(252)) if tf_vol > 0 else 0,
                    'max_drawdown': getattr(self.portfolio, f'max_drawdown_{tf}', 0)
                }
            
            # 3. Signal Consistency Analysis
            signal_strength = self._analyze_signal_consistency()
            
            # 4. Position Analysis
            position_metrics = self._analyze_positions()
            
            # 5. Risk Metrics
            risk_metrics = {
                'max_drawdown': getattr(self.portfolio, 'max_drawdown', 0),
                'calmar_ratio': getattr(self.portfolio, 'calmar_ratio', 0),
                'value_at_risk': self._calculate_var(),
                'expected_shortfall': self._calculate_expected_shortfall()
            }
            
            # 6. Combine all metrics for reward calculation
            reward_components = {
                **{'tf_' + k: v for tf, metrics in timeframe_metrics.items() for k, v in metrics.items()},
                **{'signal_' + k: v for k, v in signal_strength.items()},
                **{'pos_' + k: v for k, v in position_metrics.items()},
                **{'risk_' + k: v for k, v in risk_metrics.items()},
                'portfolio_value': portfolio_value,
                'current_pnl': current_pnl,
                'total_return': (portfolio_value / self.portfolio.initial_value - 1) if self.portfolio.initial_value > 0 else 0
            }
            
            # 1. Calculate base reward using the reward calculator with all components
            base_reward = self.reward_calculator.calculate_reward(**reward_components)
            
            # 2. Calculate chunk-based performance bonus
            chunk_bonus = self._calculate_chunk_bonus(portfolio_value)
            
            # 3. Calculate signal consistency bonus/penalty
            signal_bonus = signal_strength.get('consistency_score', 0) * 0.1  # Scale factor
            
            # 4. Apply drawdown penalty
            drawdown_penalty = self._calculate_drawdown_penalty(risk_metrics['max_drawdown'])
            
            # 5. Calculate final reward with all components
            final_reward = (
                base_reward * 0.7 +  # Base performance
                chunk_bonus * 0.2 +  # Chunk performance
                signal_bonus * 0.1 -  # Signal consistency
                drawdown_penalty      # Risk penalty
            )
            
            # 6. Update reward history with detailed metrics
            self._update_reward_history(
                portfolio_value=portfolio_value,
                base_reward=base_reward,
                chunk_bonus=chunk_bonus,
                signal_bonus=signal_bonus,
                drawdown_penalty=drawdown_penalty,
                final_reward=final_reward,
                metrics={
                    'timeframe_metrics': timeframe_metrics,
                    'signal_strength': signal_strength,
                    'position_metrics': position_metrics,
                    'risk_metrics': risk_metrics
                }
            )
            
            return final_reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            # Return a small negative reward on error to encourage exploration
            return -0.1

    def _analyze_signal_consistency(self) -> Dict[str, float]:
        """
        Analyze signal consistency across different timeframes.
        
        Returns:
            Dict containing signal consistency metrics:
            - consistency_score: Overall consistency score (0-1)
            - direction_agreement: Percentage of timeframes agreeing on direction
            - strength_score: Average signal strength
        """
        if not hasattr(self, 'state_builder'):
            return {'consistency_score': 0.5, 'direction_agreement': 0.5, 'strength_score': 0.5}
            
        signals = {}
        timeframes = getattr(self.state_builder, 'timeframes', ['5m', '1h', '4h'])
        
        # Get signals from state builder or indicators
        for tf in timeframes:
            # This is a simplified example - replace with actual signal extraction
            signals[tf] = {
                'direction': np.random.choice([-1, 0, 1]),  # Replace with real signal
                'strength': np.random.random()  # Replace with real strength
            }
        
        # Calculate consistency metrics
        directions = [s['direction'] for s in signals.values() if s['direction'] != 0]
        if not directions:
            return {'consistency_score': 0.5, 'direction_agreement': 0.5, 'strength_score': 0.5}
            
        direction_agreement = max(
            sum(1 for d in directions if d == 1) / len(directions),
            sum(1 for d in directions if d == -1) / len(directions)
        )
        
        avg_strength = np.mean([s['strength'] for s in signals.values()])
        consistency_score = direction_agreement * avg_strength
        
        return {
            'consistency_score': float(consistency_score),
            'direction_agreement': float(direction_agreement),
            'strength_score': float(avg_strength)
        }
        
    def _analyze_positions(self) -> Dict[str, float]:
        """
        Analyze current positions and calculate position-based metrics.
        
        Returns:
            Dict containing position metrics:
            - concentration: Portfolio concentration index (0-1)
            - diversification: Number of positions / max possible
            - avg_position_size: Average position size as % of portfolio
        """
        if not hasattr(self, 'portfolio') or not hasattr(self.portfolio, 'positions'):
            return {'concentration': 0.0, 'diversification': 1.0, 'avg_position_size': 0.0}
            
        positions = self.portfolio.positions
        if not positions:
            return {'concentration': 0.0, 'diversification': 1.0, 'avg_position_size': 0.0}
            
        position_sizes = [
            pos.current_value / self.portfolio.portfolio_value 
            for pos in positions.values() 
            if pos.amount != 0
        ]
        
        if not position_sizes:
            return {'concentration': 0.0, 'diversification': 1.0, 'avg_position_size': 0.0}
            
        # Herfindahl-Hirschman Index for concentration
        hhi = sum(s**2 for s in position_sizes)
        
        return {
            'concentration': float(hhi),
            'diversification': len(position_sizes) / len(positions) if positions else 0.0,
            'avg_position_size': float(np.mean(position_sizes) if position_sizes else 0.0)
        }
    
    def _calculate_var(self, confidence_level: float = 0.95, window: int = 252) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Args:
            confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
            window: Lookback window in days
            
        Returns:
            Value at Risk as a percentage of portfolio value
        """
        if not hasattr(self, 'portfolio') or not hasattr(self.portfolio, 'returns'):
            return 0.0
            
        returns = np.array(self.portfolio.returns[-window:])
        if len(returns) < 2:
            return 0.0
            
        return float(np.percentile(returns, (1 - confidence_level) * 100))
    
    def _calculate_expected_shortfall(self, confidence_level: float = 0.95, window: int = 252) -> float:
        """
        Calculate Expected Shortfall (CVaR) for the portfolio.
        
        Args:
            confidence_level: Confidence level for ES (e.g., 0.95 for 95%)
            window: Lookback window in days
            
        Returns:
            Expected Shortfall as a percentage of portfolio value
        """
        if not hasattr(self, 'portfolio') or not hasattr(self.portfolio, 'returns'):
            return 0.0
            
        returns = np.array(self.portfolio.returns[-window:])
        if len(returns) < 2:
            return 0.0
            
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return float(np.mean(returns[returns <= var]))
    
    def _calculate_chunk_bonus(self, current_value: float) -> float:
        """
        Calculate performance bonus based on chunk metrics.
        
        Args:
            current_value: Current portfolio value
            
        Returns:
            Bonus value (can be negative)
        """
        if not hasattr(self, 'chunk_metrics') or not self.chunk_metrics:
            return 0.0
            
        chunk_pnl = current_value - self.chunk_metrics['start_value']
        chunk_optimal_pnl = self.chunk_metrics.get('optimal_pnl', 0.0)
        
        if chunk_optimal_pnl <= 0:
            return 0.0
            
        performance_ratio = min(1.0, chunk_pnl / chunk_optimal_pnl)
        chunk_bonus = performance_ratio * self.chunk_metrics.get('max_bonus', 0.0)
        
        logger.debug(
            f"Chunk {self.current_chunk_idx}: PnL={chunk_pnl:.2f}, "
            f"Optimal={chunk_optimal_pnl:.2f}, "
            f"Performance={performance_ratio:.2f}, "
            f"Bonus={chunk_bonus:.4f}"
        )
        
        return float(chunk_bonus)
    
    def _calculate_drawdown_penalty(self, max_drawdown: float) -> float:
        """
        Calculate penalty based on maximum drawdown.
        
        Args:
            max_drawdown: Maximum drawdown as a decimal (e.g., 0.1 for 10%)
            
        Returns:
            Penalty value (always positive)
        """
        if max_drawdown <= 0.05:  # No penalty for <5% drawdown
            return 0.0
        elif max_drawdown > 0.20:  # Max penalty for >20% drawdown
            return 1.0
        else:
            # Linear scaling between 5% and 20% drawdown
            return (max_drawdown - 0.05) / 0.15
    
    def _update_reward_history(
        self, 
        portfolio_value: float,
        base_reward: float,
        chunk_bonus: float,
        signal_bonus: float,
        drawdown_penalty: float,
        final_reward: float,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Update reward history with detailed metrics.
        
        Args:
            portfolio_value: Current portfolio value
            base_reward: Base reward from reward calculator
            chunk_bonus: Bonus from chunk performance
            signal_bonus: Bonus from signal consistency
            drawdown_penalty: Penalty from drawdown
            final_reward: Final calculated reward
            metrics: Dictionary containing all metrics
        """
        if not hasattr(self, 'reward_history'):
            self.reward_history = []
            
        entry = {
            'step': self.current_step,
            'chunk': self.current_chunk_idx,
            'portfolio_value': portfolio_value,
            'base_reward': base_reward,
            'chunk_bonus': chunk_bonus,
            'signal_bonus': signal_bonus,
            'drawdown_penalty': drawdown_penalty,
            'total_reward': final_reward,
            'timestamp': self._get_current_timestamp().isoformat(),
            'metrics': metrics
        }
        
        self.reward_history.append(entry)
        
        # Log reward details periodically
        if self.current_step % 100 == 0:
            logger.info(
                f"Step {self.current_step}: "
                f"Portfolio={portfolio_value:.2f}, "
                f"Base={base_reward:.4f}, "
                f"Chunk={chunk_bonus:+.4f}, "
                f"Signal={signal_bonus:+.4f}, "
                f"Penalty={-drawdown_penalty:.4f}, "
                f"Total={final_reward:.4f}"
            )

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation for all assets.
        
        This method constructs a 3D observation tensor for each asset using the StateBuilder.
        The observation has shape (num_timeframes, window_size, num_features) where:
        - num_timeframes: Number of timeframes (e.g., 3 for 5m, 1h, 4h)
        - window_size: Number of time steps in the lookback window
        - num_features: Number of features per timeframe
        
        For each asset, the method:
        1. Extracts the relevant data window for each timeframe
        2. Combines data from all timeframes into a single DataFrame
        3. Adds portfolio state information if enabled
        4. Builds a 3D observation tensor using the StateBuilder
        5. Validates the observation shape
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping asset names to their 3D observation tensors
            
        Raises:
            RuntimeError: If no data is available for any asset or if the StateBuilder is not properly initialized
        """
        if not hasattr(self, 'current_data') or not self.current_data:
            raise RuntimeError("No data available in current chunk")
            
        if not hasattr(self, 'state_builder') or not self.state_builder:
            raise RuntimeError("StateBuilder not initialized")
            
        observation = {}
        
        for asset, tf_data in self.current_data.items():
            if not tf_data:
                logger.warning(f"No timeframe data available for asset {asset}")
                continue
                
            try:
                # Log the current step and chunk for debugging
                logger.debug(
                    f"Building observation for {asset} at step {self.step_in_chunk}/"
                    f"{max(len(df) for df in tf_data.values() if df is not None)} in chunk {self.current_chunk_idx}"
                )
                
                # Get the current window of data for each timeframe
                combined_data = []
                
                for tf, df in tf_data.items():
                    if df is None or df.empty:
                        logger.warning(f"No data for {asset} on {tf} timeframe")
                        continue
                        
                    # Get the current window of data (up to current step)
                    window_start = max(0, self.step_in_chunk - self.state_builder.window_size + 1)
                    window_end = min(self.step_in_chunk + 1, len(df))
                    
                    if window_start >= window_end:
                        logger.warning(
                            f"Invalid window for {asset} {tf}: start={window_start}, end={window_end}, "
                            f"data_length={len(df)}"
                        )
                        continue
                    
                    # Get the data slice for this window
                    data_slice = df.iloc[window_start:window_end].copy()
                    
                    # Add prefix to column names to indicate timeframe
                    data_slice = data_slice.add_prefix(f"{tf}_")
                    
                    # Ensure we have a timestamp column for merging
                    if 'timestamp' in data_slice.columns:
                        data_slice = data_slice.rename(columns={f"{tf}_timestamp": 'timestamp'})
                    
                    combined_data.append(data_slice)
                
                if not combined_data:
                    logger.warning(f"No valid data for {asset} across any timeframe")
                    continue
                
                # Merge data from all timeframes on timestamp
                if len(combined_data) > 1:
                    # Start with the first dataframe
                    merged_data = combined_data[0]
                    
                    # Merge with remaining dataframes
                    for df in combined_data[1:]:
                        if 'timestamp' in merged_data.columns and 'timestamp' in df.columns:
                            merged_data = pd.merge(
                                merged_data, 
                                df, 
                                on='timestamp', 
                                how='outer',
                                suffixes=('', '_dup')
                            )
                            
                            # Handle any duplicate columns from the merge
                            dup_cols = [col for col in merged_data.columns if col.endswith('_dup')]
                            if dup_cols:
                                logger.warning(f"Duplicate columns after merge for {asset}: {dup_cols}")
                                merged_data = merged_data.drop(columns=dup_cols)
                        else:
                            logger.warning("Missing timestamp column for merge")
                            merged_data = pd.concat([merged_data, df], axis=1)
                else:
                    merged_data = combined_data[0]
                
                # Sort by timestamp to ensure correct order
                if 'timestamp' in merged_data.columns:
                    merged_data = merged_data.sort_values('timestamp')
                
                # Fill any NaN values with zeros (or another appropriate fill method)
                merged_data = merged_data.fillna(0.0)
                
                # Log the timestamp range for this observation
                if not merged_data.empty and 'timestamp' in merged_data.columns:
                    logger.debug(
                        f"Data window for {asset}: {merged_data['timestamp'].iloc[0]} to "
                        f"{merged_data['timestamp'].iloc[-1]} ({len(merged_data)} steps)"
                    )
                
                # Add portfolio state to the observation if enabled
                if hasattr(self, 'include_portfolio_state') and self.include_portfolio_state:
                    self._add_portfolio_state_to_observation(merged_data)
                
                # Build the 3D observation using the StateBuilder
                obs_tensor = self.state_builder.build_observation(merged_data)
                
                # Validate the observation shape
                expected_shape = self.state_builder.get_observation_shape()
                if obs_tensor.shape != expected_shape:
                    logger.warning(
                        f"Unexpected observation shape for {asset}: {obs_tensor.shape} "
                        f"(expected {expected_shape})"
                    )
                
                # Log observation details for debugging
                if self.current_step % 100 == 0:
                    logger.info(
                        f"Observation for {asset}: shape={obs_tensor.shape}, "
                        f"min={obs_tensor.min():.4f}, max={obs_tensor.max():.4f}, "
                        f"mean={obs_tensor.mean():.4f}, NaN={np.isnan(obs_tensor).any()}"
                    )
                
                observation[asset] = obs_tensor
                
            except Exception as e:
                logger.error(
                    f"Error building observation for {asset}: {str(e)}",
                    exc_info=True
                )
                
                # Return a zero observation with correct shape if possible
                if hasattr(self.state_builder, 'get_observation_shape'):
                    obs_shape = self.state_builder.get_observation_shape()
                    observation[asset] = np.zeros(obs_shape, dtype=np.float32)
                    logger.warning(
                        f"Using zero observation for {asset} due to error. "
                        f"Shape: {obs_shape}"
                    )
                else:
                    # If we can't get the shape, re-raise the exception
                    raise RuntimeError(
                        f"Failed to build observation for {asset} and cannot create fallback: {str(e)}"
                    )
        
        if not observation:
            raise RuntimeError(
                f"Failed to build observations for any asset in chunk {self.current_chunk_idx} "
                f"at step {self.step_in_chunk}"
            )
            
        return observation

    def _add_portfolio_state_to_observation(self, data_slice: pd.DataFrame) -> None:
        """
        Add portfolio state information to the observation data.
        
        This method enriches the market data with portfolio-related features
        like current positions, cash balance, and portfolio metrics.
        
        Args:
            data_slice: DataFrame containing the current window of market data
        """
        if not hasattr(self, 'portfolio') or not hasattr(self.portfolio, 'positions'):
            return
            
        # Get the last row of the data (current timestep)
        last_row = data_slice.iloc[-1:].copy()
        
        # Add portfolio state features
        data_slice['portfolio_cash'] = self.portfolio.cash
        data_slice['portfolio_value'] = self.portfolio.portfolio_value
        data_slice['portfolio_returns'] = self.portfolio.returns
        data_slice['portfolio_sharpe'] = getattr(self.portfolio, 'sharpe_ratio', 0.0)
        data_slice['portfolio_drawdown'] = getattr(self.portfolio, 'max_drawdown', 0.0)
        
        # Add position information for each asset
        for asset, position in self.portfolio.positions.items():
            pos_col = f'position_{asset}'
            data_slice[pos_col] = position.amount
            
            # Add position value and PnL if available
            if hasattr(position, 'value'):
                data_slice[f'{pos_col}_value'] = position.value
            if hasattr(position, 'unrealized_pnl'):
                data_slice[f'{pos_col}_pnl'] = position.unrealized_pnl
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment state.
        
        Returns:
            Dictionary containing diagnostic information
        """
        info = {
            'step': self.current_step,
            'chunk_idx': self.current_chunk_idx,
            'step_in_chunk': self.step_in_chunk,
            'portfolio_value': float(self.portfolio.get_portfolio_value()),
            'cash': float(self.portfolio.cash),
            'positions': {k: float(v) for k, v in self.portfolio.positions.items()},
            'current_prices': self._get_current_prices(),
            'optimal_chunk_pnl': float(self.optimal_chunk_pnl) if hasattr(self, 'optimal_chunk_pnl') else 0.0,
        }
        
        # Add last chunk's PnL and bonus if available
        if hasattr(self, 'last_chunk_pnl'):
            info.update({
                'last_chunk_pnl': float(self.last_chunk_pnl),
                'last_optimal_bonus': float(self.last_optimal_bonus)
            })
            
        return info

    def _get_current_timestamp(self) -> pd.Timestamp:
        """
        Get the current timestamp.
        
        Returns:
            pd.Timestamp: The current timestamp
            
        Raises:
            RuntimeError: If no data is available or if the current step is out of bounds
        """
        if not hasattr(self, 'current_data') or not self.current_data:
            raise RuntimeError("No data available in current chunk")
            
        # Get the first available asset's data
        sample_asset = next(iter(self.current_data.keys()))
        if sample_asset not in self.current_data or not self.current_data[sample_asset]:
            raise RuntimeError(f"No timeframe data available for asset {sample_asset}")
            
        # Get the highest frequency timeframe (first in the list)
        highest_tf = self.state_builder.timeframes[0] if hasattr(self, 'state_builder') and hasattr(self.state_builder, 'timeframes') and self.state_builder.timeframes else None
        
        # Try to get data from the highest frequency timeframe first
        if highest_tf and highest_tf in self.current_data[sample_asset] and self.current_data[sample_asset][highest_tf] is not None:
            df = self.current_data[sample_asset][highest_tf]
            if not df.empty and self.step_in_chunk < len(df):
                if hasattr(df, 'index') and hasattr(df.index, '__getitem__'):
                    return df.index[self.step_in_chunk]
                elif hasattr(df, 'iloc') and hasattr(df.iloc, '__getitem__'):
                    return df.iloc[self.step_in_chunk].name
        
        # Fall back to any available timeframe
        for tf, df in self.current_data[sample_asset].items():
            if df is not None and not df.empty and self.step_in_chunk < len(df):
                if hasattr(df, 'index') and hasattr(df.index, '__getitem__'):
                    return df.index[self.step_in_chunk]
                elif hasattr(df, 'iloc') and hasattr(df.iloc, '__getitem__'):
                    return df.iloc[self.step_in_chunk].name
        
        # If we get here, we couldn't find a valid timestamp
        raise RuntimeError(
            f"Could not determine timestamp for asset {sample_asset} at step {self.step_in_chunk}. "
            f"Available timeframes: {list(self.current_data[sample_asset].keys()) if sample_asset in self.current_data else 'none'}"
        )
        
    def _get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all assets.
        
        This method retrieves the most recent 'close' price for each asset
        from the highest frequency timeframe (typically '5m').
        
        Returns:
            Dict[str, float]: Mapping of asset names to their current prices
            
        Raises:
            RuntimeError: If no data is available or if the price data is invalid
        """
        if not hasattr(self, 'current_data') or not self.current_data:
            raise RuntimeError("No data available to get current prices")
            
        prices = {}
        highest_tf = self.state_builder.timeframes[0]  # First timeframe is highest frequency
        
        for asset, tf_data in self.current_data.items():
            if not tf_data or highest_tf not in tf_data or tf_data[highest_tf] is None or tf_data[highest_tf].empty:
                logger.warning(f"No {highest_tf} data available for asset {asset}")
                continue
                
            df = tf_data[highest_tf]
            if self.step_in_chunk >= len(df):
                logger.warning(f"Step {self.step_in_chunk} out of bounds for {asset} {highest_tf} data (length {len(df)})")
                continue
                
            try:
                # Get the close price for the current step from the highest frequency timeframe
                close_col = 'close'  # Column name without timeframe prefix in the per-timeframe DataFrames
                if close_col not in df.columns:
                    # Try to find a close column with any suffix
                    close_cols = [col for col in df.columns if 'close' in col.lower()]
                    if not close_cols:
                        raise ValueError(f"No close price column found for {asset} in {highest_tf} timeframe")
                    close_col = close_cols[0]
                    
                price = df[close_col].iloc[self.step_in_chunk]
                if pd.isna(price):
                    raise ValueError(f"Price is NaN for {asset} at step {self.step_in_chunk}")
                    
                prices[asset] = float(price)
                
                # Log the first few prices for debugging
                if len(prices) <= 3:  # Only log first few to avoid flooding logs
                    logger.debug(f"Current price for {asset}: {price:.8f} ({close_col})")
                    
            except Exception as e:
                logger.error(f"Error getting price for {asset}: {str(e)}")
                # Skip this asset but continue with others
                continue
                
        if not prices:
            raise RuntimeError("Failed to get prices for any asset")
            
        return prices
        
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
            
        Returns:
            Optional[str]: String representation if mode is 'ansi', None otherwise
        """
        if mode == 'human':
            print(f"Step: {self.current_step} (Chunk {self.current_chunk_idx}, Step {self.step_in_chunk})")
            print(f"Portfolio Value: {self.portfolio.get_portfolio_value():.2f} | "
                  f"Cash: {self.portfolio.cash:.2f} | "
                  f"Positions: {self.portfolio.positions}")
            
            if hasattr(self, 'last_chunk_pnl'):
                print(f"Last Chunk PnL: {self.last_chunk_pnl:.2f} | "
                      f"Optimal Bonus: {self.last_optimal_bonus:.4f}")
            return None
        
        if mode == 'ansi':
            # Return a string instead of printing
            output = [
                f"Step: {self.current_step} (Chunk {self.current_chunk_idx}, Step {self.step_in_chunk})",
                f"Portfolio Value: {self.portfolio.get_portfolio_value():.2f} | "
                f"Cash: {self.portfolio.cash:.2f} | "
                f"Positions: {self.portfolio.positions}"
            ]
            
            if hasattr(self, 'last_chunk_pnl'):
                output.extend([
                    f"Last Chunk PnL: {self.last_chunk_pnl:.2f}",
                    f"Optimal Bonus: {self.last_optimal_bonus:.4f}"
                ])
                
            return "\n".join(output)
        
        return None
