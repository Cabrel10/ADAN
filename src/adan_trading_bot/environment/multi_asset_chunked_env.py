#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-asset trading environment with chunked data loading.

This environment extends the base trading environment to support multiple assets
and efficient data loading using ChunkedDataLoader.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path

from ..data_processing.chunked_loader import ChunkedDataLoader
from ..environment.state_builder import StateBuilder
from ..environment.reward_calculator import RewardCalculator
from ..portfolio.portfolio_manager import PortfolioManager
from ..trading.order_manager import OrderManager

logger = logging.getLogger(__name__)

class MultiAssetChunkedEnv(gym.Env):
    """
    A multi-asset trading environment with efficient chunked data loading.
    
    This environment loads data in chunks to manage memory usage and supports
    trading multiple assets simultaneously.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-asset trading environment.
        
        Args:
            config: Configuration dictionary containing:
                - data: Data loading configuration
                - env: Environment parameters
                - portfolio: Portfolio management settings
                - trading: Trading parameters
        """
        super().__init__()
        self.config = config
        
        # Initialize core components
        self._initialize_components()
        
        # Set up action and observation spaces
        self._setup_spaces()
        
        # Initialize state
        self.current_step = 0
        self.current_chunk = 0
        self.current_data = {}
        self.done = False
        
        logger.info("MultiAssetChunkedEnv initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all environment components."""
        # 1. Initialize data loader
        self._init_data_loader()
        
        # 2. Initialize portfolio manager
        self.portfolio = PortfolioManager(
            initial_balance=self.config['portfolio'].get('initial_balance', 10000.0),
            max_leverage=self.config['portfolio'].get('max_leverage', 3.0),
            risk_per_trade=self.config['portfolio'].get('risk_per_trade', 0.01)
        )
        
        # 3. Initialize order manager
        self.order_manager = OrderManager(
            slippage=self.config['trading'].get('slippage', 0.0005),
            commission=self.config['trading'].get('commission', 0.0005)
        )
        
        # 4. Initialize reward calculator
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
        
        self.data_loader = ChunkedDataLoader(
            file_path=Path(data_config.get('data_dir', 'data/final')),
            chunk_size=data_config.get('chunk_size', 10000),
            assets_list=data_config.get('assets'),
            features_by_timeframe=self._get_features_config()
        )
        
        # Store assets list for easy access
        self.assets = self.data_loader.assets_list
        if not self.assets:
            raise ValueError("No assets available for trading")
            
        logger.info(f"Initialized data loader with {len(self.assets)} assets")
    
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
        
        # Observation space will be set after seeing the first data point
        self.observation_space = None
    
    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Reset environment state
        self.current_step = 0
        self.current_chunk = 0
        self.done = False
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Load the first chunk of data
        self.current_data = self.data_loader.load_chunk(0)
        
        # Initialize observation space if not already set
        if self.observation_space is None:
            self._initialize_observation_space()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _initialize_observation_space(self) -> None:
        """Initialize the observation space based on the first data point."""
        if not self.current_data:
            raise ValueError("No data available to initialize observation space")
        
        # Get state shape from the state builder
        sample_asset = next(iter(self.current_data.keys()))
        sample_data = self.current_data[sample_asset].iloc[0:1]
        state_shape = self.state_builder.get_state_shape(sample_data)
        
        # Create observation space
        self.observation_space = spaces.Dict({
            asset: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=state_shape,
                dtype=np.float32
            )
            for asset in self.assets
        })
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Action to take (one per asset)
            
        Returns:
            observation: Next observation
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
        
        # Check if we need to load the next chunk
        if self.current_step >= len(next(iter(self.current_data.values()))):
            self.current_chunk += 1
            
            if self.current_chunk >= len(self.data_loader):
                self.done = True
            else:
                self.current_data = self.data_loader.load_chunk(self.current_chunk)
                self.current_step = 0
        
        # Get next observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self.done
        truncated = False  # Can be set by wrappers
        
        # Get additional info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_trades(self, action: np.ndarray) -> None:
        """Execute trades based on the action vector."""
        if len(action) != len(self.assets):
            raise ValueError(f"Action dimension {len(action)} does not match number of assets {len(self.assets)}")
        
        current_prices = self._get_current_prices()
        
        for i, (asset, action_idx) in enumerate(zip(self.assets, action)):
            if action_idx == 0:  # Hold
                continue
                
            price = current_prices[asset]
            
            if action_idx == 1:  # Buy
                self.order_manager.enter_trade(
                    asset=asset,
                    price=price,
                    size=self.portfolio.calculate_position_size(price),
                    direction='long',
                    timestamp=self._get_current_timestamp()
                )
            elif action_idx == 2:  # Sell
                self.order_manager.exit_trade(
                    asset=asset,
                    price=price,
                    timestamp=self._get_current_timestamp()
                )
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all assets."""
        return {
            asset: df.iloc[self.current_step]['5m_close']
            for asset, df in self.current_data.items()
            if not df.empty and self.current_step < len(df)
        }
    
    def _get_current_timestamp(self) -> pd.Timestamp:
        """Get the current timestamp."""
        sample_asset = next(iter(self.current_data.keys()))
        return self.current_data[sample_asset].index[self.current_step]
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        observation = {}
        
        for asset, df in self.current_data.items():
            if self.current_step < len(df):
                current_data = df.iloc[max(0, self.current_step - self.state_builder.window_size + 1):self.current_step + 1]
                observation[asset] = self.state_builder.build_state(current_data)
        
        return observation
    
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current step."""
        # Get portfolio value change
        portfolio_value = self.portfolio.get_portfolio_value()
        reward = self.reward_calculator.calculate_reward(
            current_value=portfolio_value,
            previous_value=self.portfolio.previous_value,
            max_drawdown=self.portfolio.max_drawdown,
            sharpe_ratio=self.portfolio.sharpe_ratio,
            sortino_ratio=self.portfolio.sortino_ratio
        )
        
        # Update portfolio metrics
        self.portfolio.update_metrics(portfolio_value)
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            'step': self.current_step,
            'chunk': self.current_chunk,
            'portfolio_value': self.portfolio.get_portfolio_value(),
            'cash': self.portfolio.cash,
            'positions': self.portfolio.positions,
            'current_prices': self._get_current_prices()
        }
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step}, "
                  f"Portfolio Value: {self.portfolio.get_portfolio_value():.2f}, "
                  f"Cash: {self.portfolio.cash:.2f}, "
                  f"Positions: {self.portfolio.positions}")
    
    def close(self) -> None:
        """Clean up resources."""
        pass  # No special cleanup needed
