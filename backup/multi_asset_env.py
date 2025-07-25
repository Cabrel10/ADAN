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
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging

from adan_trading_bot.data.chunked_data_loader import ChunkedDataLoader
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from adan_trading_bot.orders.order_manager import OrderManager
from adan_trading_bot.rewards.reward_calculator import RewardCalculator
from adan_trading_bot.state.state_builder import StateBuilder

logger = logging.getLogger(__name__)

class AdanTradingEnv(gym.Env):
    """A comprehensive, multi-asset trading environment for reinforcement learning.

    This environment integrates all the components of the trading system:
    - Data Loading: Uses ChunkedDataLoader to get data in chunks for memory efficiency.
    - State Management: Uses StateBuilder to construct 3D observation space.
    - Portfolio Management: Uses PortfolioManager to track finances across multiple assets.
    - Order Execution: Uses OrderManager to simulate trades with fees and slippage.
    - Reward Calculation: Uses RewardCalculator for reward shaping with optimal trade bonus.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data_loader: ChunkedDataLoader, config: Dict[str, Any], mode: str = None):
        """
        Initialize the environment with a ChunkedDataLoader.

        Args:
            data_loader: Pre-initialized ChunkedDataLoader instance for the current split
            config: Configuration dictionary with sections for data, portfolio, trading, etc.
            mode: Either 'train', 'val', or 'test' to determine which data split to use.
                 If None, will be extracted from config or default to 'train'.
        """
        super().__init__()
        self.config = config
        self.mode = mode if mode is not None else config.get('mode', 'train')
        
        # Store the data loader
        self.data_loader = data_loader
        
        # Initialize components
        self._init_components()
        
        # Initialize state tracking
        self.current_step = 0
        self.step_in_chunk = 0
        self.current_chunk_idx = 0
        self.current_chunk = None
        self.optimal_chunk_pnl = 0.0
        self.done = False
        
        logger.info(f"AdanTradingEnv initialized in {self.mode} mode with {len(self.data_loader.assets_list)} assets")
    
    def _init_components(self) -> None:
        """
        Initialize all environment components with robust error handling.
        """
        try:
            logger.info("Initializing environment components...")
            
            # Initialize data loader
            logger.debug("Initializing data loader...")
            self._init_data_loader()
            
            # Get features configuration
            logger.debug("Getting features configuration...")
            features_config = self._get_features_config()
            
            # Initialize state builder
            logger.debug("Initializing state builder...")
            self.state_builder = StateBuilder(
                features_config=features_config,
                include_portfolio_state=self.config.get('include_portfolio_state', True)
            )
            
            # Initialize portfolio manager
            logger.debug("Initializing portfolio manager...")
            portfolio_config = self.config.get('portfolio', {})
            if not portfolio_config:
                raise ValueError("Missing portfolio configuration")
            
            required_portfolio_keys = ['initial_capital', 'risk_limit', 'max_leverage']
            missing_keys = [k for k in required_portfolio_keys if k not in portfolio_config]
            if missing_keys:
                raise ValueError(f"Missing required portfolio config keys: {missing_keys}")
                
            self.portfolio = PortfolioManager(
                initial_capital=portfolio_config['initial_capital'],
                risk_limit=portfolio_config['risk_limit'],
                max_leverage=portfolio_config['max_leverage']
            )
            
            # Initialize order manager
            logger.debug("Initializing order manager...")
            trading_config = self.config.get('trading', {})
            if not trading_config:
                raise ValueError("Missing trading configuration")
            
            required_trading_keys = ['slippage_model', 'fee_structure']
            missing_keys = [k for k in required_trading_keys if k not in trading_config]
            if missing_keys:
                raise ValueError(f"Missing required trading config keys: {missing_keys}")
                
            self.order_manager = OrderManager(
                slippage_model=trading_config['slippage_model'],
                fee_structure=trading_config['fee_structure']
            )
            
            # Initialize reward calculator
            logger.debug("Initializing reward calculator...")
            reward_config = self.config.get('reward', {})
            if not reward_config:
                raise ValueError("Missing reward configuration")
            
            self.reward_calculator = RewardCalculator(
                config=reward_config
            )
            
            # Setup action and observation spaces
            logger.debug("Setting up action and observation spaces...")
            self._setup_spaces()
            
            logger.info("All components initialized successfully")
            
        except KeyError as e:
            logger.error(f"Missing required configuration key: {str(e)}", exc_info=True)
            raise ValueError(f"Missing required configuration: {str(e)}")
        except ValueError as e:
            logger.error(f"Configuration validation error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}", exc_info=True)
            raise
    
    def _init_data_loader(self) -> None:
        """
        Store assets list from the pre-initialized data loader.
        
        This method extracts the list of available assets from the data loader
        that was passed to the environment constructor.
        """
        # Store assets list for easy access
        self.assets = self.data_loader.assets_list
        if not self.assets:
            raise ValueError("No assets available for trading")
            
        logger.info(f"Using pre-initialized {self.mode} data loader with {len(self.assets)} assets")
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
        self.action_space = spaces.MultiDiscrete([3] * len(self.assets))
        obs_shape = self.state_builder.get_observation_shape()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            - observation: Initial observation for all assets
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
                raise RuntimeError("PortfolioManager not initialized")
            
            self.portfolio.reset()
            
            # Load first chunk of data
            self.current_chunk = self.data_loader.get_chunk(0)
            if self.current_chunk is None:
                raise RuntimeError("No data available for chunk 0")
            
            # Get first asset's data to log chunk details
            first_asset = next(iter(self.current_chunk))
            first_df = self.current_chunk[first_asset]['5m']
            
            # If the index is a DatetimeIndex, use it directly
            if isinstance(first_df.index, pd.DatetimeIndex):
                chunk_start = first_df.index[0]
                chunk_end = first_df.index[-1]
            else:
                # Si l'index n'est pas un DatetimeIndex, on utilise les premières et dernières lignes
                chunk_start = first_df.iloc[0].name if first_df.index.name else "unknown"
                chunk_end = first_df.iloc[-1].name if first_df.index.name else "unknown"
            
            num_steps = len(first_df)
            
            logger.info(
                f"Reset environment. Loaded chunk 0 with data from {chunk_start} to {chunk_end} "
                f"({num_steps} steps)"
            )
            
            # Initialize chunk metrics
            self._initialize_chunk_metrics()
            
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

    def _calculate_position_size(self, asset: str, price: float) -> float:
        """
        Calculate the position size for a given asset based on available capital.
        
        Args:
            asset: Asset symbol
            price: Current price of the asset
            
        Returns:
            float: Calculated position size
            
        Raises:
            ValueError: If price is invalid or configuration is missing
        """
        try:
            if price <= 0:
                logger.warning(f"Invalid price ({price}) for {asset}, returning 0.0 position size")
                return 0.0
            
            # Get position size configuration
            config = self.config.get('position_size', {})
            min_position = config.get('min_position', 0.0)
            max_position = config.get('max_position', float('inf'))
            position_factor = config.get('position_factor', 1.0)
            
            # Calculate available capital
            available_capital = self.portfolio.cash
            if available_capital <= 0:
                logger.warning(f"No available capital for {asset}, returning 0.0 position size")
                return 0.0
            
            # Calculate position size as a fraction of available capital
            position_size = available_capital * position_factor
            
            # Apply price-based adjustment
            if price > 0:
                position_size = min(position_size, available_capital / price)
            
            # Apply min/max constraints
            position_size = max(min_position, min(max_position, position_size))
            
            # Calculate position size factor
            position_size_factor = position_size / available_capital if available_capital > 0 else 0.0
            
            # Log position size calculation
            logger.debug(
                f"Calculated position size for {asset}: "
                f"price={price:.8f}, "
                f"available_capital={available_capital:.2f}, "
                f"position_size={position_size:.6f}, "
                f"factor={position_size_factor:.2f}")
            
            return position_size
            
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            raise ValueError(f"Missing required configuration: {e}")
        except Exception as e:
            logger.error(f"Error calculating position size for {asset}: {e}", exc_info=True)
            raise

    def _execute_trades(self, action: np.ndarray) -> None:
        """
        Execute trades based on the action vector.

        This method processes each action in the action vector and executes the corresponding
        trade (buy/sell) for each asset. It handles errors gracefully and logs all actions.

        Args:
            action: Array of actions (0=hold, 1=buy, 2=sell) for each asset

        Raises:
            ValueError: If action vector has incorrect shape or values
        """
        try:
            # Validate action shape
            if action.shape != (len(self.assets),):
                raise ValueError(
                    f"Action shape mismatch: expected {(len(self.assets),)}, got {action.shape}"
                )

            # Execute trades for each asset
            current_timestamp = self._get_current_timestamp()
            current_prices = self._get_current_prices()

            for i, asset in enumerate(self.assets):
                try:
                    # Get current price
                    if asset not in current_prices:
                        logger.warning(f"No price available for asset {asset}, skipping trade")
                        continue

                    price = current_prices[asset]

                    # Log the action for debugging
                    action_str = {1: 'BUY', 2: 'SELL'}.get(int(action[i]), 'HOLD')
                    logger.debug(
                        f"Executing {action_str} order for {asset} at price {price:.8f} "
                        f"(Step {self.current_step}, Chunk {self.current_chunk_idx})"
                    )

                    if action[i] == 1:  # Buy
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
                    elif action[i] == 2:  # Sell
                        current_position = self.portfolio.positions.get(asset, 0)
                        if current_position > 0:
                            self.order_manager.exit_trade(
                                asset=asset,
                                price=price,
                                size=current_position,
                                direction='long',
                                timestamp=current_timestamp
                            )
                            logger.info(
                                f"Exited long position: {asset} x{current_position:.6f} @ {price:.8f} "
                                f"(Value: {current_position * price:.2f})"
                            )
                    else:
                        continue

                except Exception as e:
                    logger.error(
                        f"Error executing trade for {asset}: {str(e)}",
                        exc_info=True
                    )
                    # Continue with other assets even if one fails
                    continue

            # Update portfolio state after all trades
            self.portfolio.update_state()

        except Exception as e:
            logger.error(f"Error in _execute_trades: {str(e)}", exc_info=True)
            raise

    def _get_current_timestamp(self) -> pd.Timestamp:
        """
        Get the current timestamp from the highest frequency timeframe.

        Returns:
            Current timestamp as pandas Timestamp
        """
        try:
            if not hasattr(self, 'current_data') or not self.current_data:
                raise RuntimeError("No data available in current chunk")

            # Get the highest frequency timeframe (first in the list)
            highest_tf = self.state_builder.timeframes[0] if hasattr(self.state_builder, 'timeframes') else '5m'

            # Get current row for each asset
            current_timestamp = None

            for asset, data_by_tf in self.current_data.items():
                if highest_tf in data_by_tf and not data_by_tf[highest_tf].empty:
                    timestamp = data_by_tf[highest_tf].iloc[self.step_in_chunk]['timestamp']
                    if current_timestamp is None or timestamp > current_timestamp:
                        current_timestamp = timestamp

            if current_timestamp is None:
                raise RuntimeError("No valid timestamp found in current data")

            # Convert to pandas Timestamp if needed
            if not isinstance(current_timestamp, pd.Timestamp):
                current_timestamp = pd.Timestamp(current_timestamp)

            return current_timestamp
        except Exception as e:
            logger.error(f"Error in _get_current_timestamp: {str(e)}", exc_info=True)
            raise

    def _get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all assets.
        
        Returns:
            Dictionary mapping asset names to their current prices
        """
        try:
            if not hasattr(self, 'current_data') or not self.current_data:
                raise RuntimeError("No data available in current chunk")
            
            prices = {}
            
            # Get the highest frequency timeframe (first in the list)
            highest_tf = self.state_builder.timeframes[0] if hasattr(self.state_builder, 'timeframes') else '5m'
            
            # Get current row for each asset
            for asset, data_by_tf in self.current_data.items():
                if highest_tf in data_by_tf and not data_by_tf[highest_tf].empty:
                    price = data_by_tf[highest_tf].iloc[self.step_in_chunk]['close']
                    prices[asset] = price
            
            if not prices:
                raise RuntimeError("No valid prices found in current data")
            
            return prices
        except Exception as e:
            logger.error(f"Error in _get_current_prices: {str(e)}", exc_info=True)
            raise

    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
            
        Returns:
            Optional[str]: String representation if mode is 'ansi', None otherwise
        """
        try:
            if mode == 'ansi':
                # Get current state information
                info = self._get_info()
                
                # Create a formatted string with current state
                output = []
                output.append(f"\n=== ADAN Trading Environment State ===")
                output.append(f"Mode: {info['config']['mode']}")
                output.append(f"Step: {info['step']} ({info['step_in_chunk']}/{info['chunk_idx']})")
                output.append(f"Total Chunks: {info['total_chunks']}")
                
                # Portfolio state
                output.append("\n=== Portfolio ===")
                output.append(f"Cash: {info['portfolio']['cash']:.2f}")
                output.append(f"Total Value: {info['portfolio']['total_value']:.2f}")
                output.append(f"Returns: {info['portfolio']['returns']:.2%}")
                output.append(f"Sharpe Ratio: {info['portfolio']['sharpe_ratio']:.2f}")
                output.append(f"Max Drawdown: {info['portfolio']['max_drawdown']:.2%}")
                
                # Positions
                if info['portfolio']['positions']:
                    output.append("\n=== Positions ===")
                    for asset, size in info['portfolio']['positions'].items():
                        price = info['current_prices'].get(asset, 0.0)
                        value = size * price
                        output.append(f"{asset}: size={size:.6f}, price={price:.8f}, value={value:.2f}")
                else:
                    output.append("\nNo positions held")
                
                # Chunk metrics
                if info['chunk_metrics']:
                    output.append("\n=== Chunk Metrics ===")
                    output.append(f"Start Value: {info['chunk_metrics']['start_value']:.2f}")
                    output.append(f"Optimal PnL: {info['chunk_metrics']['optimal_pnl']:.2f}")
                    output.append(f"Max Bonus: {info['chunk_metrics']['max_bonus']:.2f}")
                    output.append(f"Size Factor: {info['chunk_metrics']['size_factor']:.2f}")
                
                return '\n'.join(output)
            else:
                return None
        except Exception as e:
            logger.error(f"Error in render: {str(e)}", exc_info=True)
            return None
