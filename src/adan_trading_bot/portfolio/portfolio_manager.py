#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Portfolio management module for the ADAN trading bot.

This module is responsible for tracking the agent's financial status, including
capital, positions, and performance metrics.
"""

from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Position:
    """Represents a single, simple trading position (long or short)."""
    def __init__(self):
        self.is_open = False
        self.entry_price = 0.0
        self.size = 0.0  # Number of units

    def open(self, entry_price: float, size: float):
        self.is_open = True
        self.entry_price = entry_price
        self.size = size

    def close(self):
        self.is_open = False
        self.entry_price = 0.0
        self.size = 0.0

    def get_status(self) -> str:
        return f"Open ({self.size:.4f} units @ {self.entry_price:.2f})" if self.is_open else "Closed"

class PortfolioManager:
    """
    Manages the trading portfolio for a single asset.

    Handles capital allocation, tracks PnL, and enforces risk rules defined
    in the environment configuration. Also tracks performance per data chunk
    for reward shaping and learning purposes.
    """
    def __init__(self, env_config: Dict[str, Any]):
        """
        Initializes the PortfolioManager.

        Args:
            env_config: The environment configuration dictionary.
        """
        self.config = env_config
        self.initial_equity = env_config.get('initial_equity', 10000.0)
        self.current_equity = self.initial_equity
        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        self.chunk_pnl: Dict[int, Dict[str, float]] = {}
        self.current_chunk_id: int = 0
        self.chunk_start_equity: float = self.initial_equity
        self.initial_capital = self.config['initial_capital']
        self.commission_pct = self.config['trading_rules']['commission_pct']
        
        self.reset()

    def reset(self):
        """Resets the portfolio to its initial state."""
        self.cash = self.initial_capital
        self.position = Position()
        self.total_capital = self.initial_capital
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_count = 0
        logger.info(f"Portfolio reset. Initial capital: {self.initial_capital:.2f}")

    def update_market_price(self, current_price: float):
        """
        Updates the portfolio's total value based on the current market price.
        """
        if self.position.is_open:
            self.unrealized_pnl = (current_price - self.position.entry_price) * self.position.size
            self.total_capital = self.cash + self.position.size * current_price
        else:
            self.unrealized_pnl = 0.0
            self.total_capital = self.cash

    def open_position(self, price: float) -> bool:
        """
        Opens a new long position.

        Args:
            price: The price at which to open the position.

        Returns:
            True if the position was opened successfully, False otherwise.
        """
        if self.position.is_open:
            logger.warning("Cannot open a new position while one is already open.")
            return False

        # Use all available cash to buy
        size_to_buy = self.cash / price
        commission = size_to_buy * price * self.commission_pct
        
        self.cash -= commission
        self.position.open(price, size_to_buy)
        self.trade_count += 1
        logger.debug(f"Opened position: {size_to_buy:.4f} units at {price:.2f}")
        return True

    def close_position(self, price: float) -> float:
        """
        Closes the current open position.

        Args:
            price: The price at which to close the position.

        Returns:
            The realized PnL from the trade.
        """
        if not self.position.is_open:
            logger.warning("Cannot close a position when none is open.")
            return 0.0

        # Calculate PnL and update cash
        trade_pnl = (price - self.position.entry_price) * self.position.size
        commission = self.position.size * price * self.commission_pct
        
        self.cash += self.position.size * price - commission
        self.realized_pnl += trade_pnl - commission
        
        logger.debug(f"Closed position at {price:.2f}. PnL: {trade_pnl - commission:.2f}")
        self.position.close()
        return trade_pnl - commission

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns a dictionary of current portfolio metrics.
        """
        total_pnl_pct = ((self.total_capital / self.initial_capital) - 1) * 100
        return {
            'initial_capital': self.initial_capital,
            'total_capital': self.total_capital,
            'cash': self.cash,
            'position_size': self.position.size,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl_pct': total_pnl_pct,
            'trade_count': self.trade_count
        }

    def get_state_features(self) -> np.ndarray:
        """
        Returns a numpy array of features representing the portfolio's state.
        """
        # [has_position, relative_pnl]
        has_position = 1.0 if self.position.is_open else 0.0
        
        if self.position.is_open:
            # PnL relative to the initial investment of that position
            entry_value = self.position.entry_price * self.position.size
            relative_pnl = self.unrealized_pnl / entry_value if entry_value != 0 else 0.0
        else:
            relative_pnl = 0.0
            
        return np.array([has_position, relative_pnl], dtype=np.float32)

    def is_bankrupt(self) -> bool:
        """
        Checks if the portfolio value has fallen below a critical threshold.
        """
        # Consider bankrupt if capital is less than 1% of initial capital
        return self.total_capital < (self.initial_capital * 0.01)
        
    def start_new_chunk(self) -> None:
        """
        Call this method when starting to process a new chunk of data.
        This will finalize the previous chunk's PnL and start tracking a new chunk.
        """
        # Finalize the previous chunk's PnL if this isn't the first chunk
        if self.current_chunk_id > 0:
            self._finalize_chunk_pnl()
            
        # Start a new chunk
        self.current_chunk_id += 1
        self.chunk_start_equity = self.total_capital
        logger.info(f"Started tracking new chunk {self.current_chunk_id} with starting equity: ${self.chunk_start_equity:.2f}")
    
    def _finalize_chunk_pnl(self) -> None:
        """Calculate and store the PnL for the current chunk."""
        if self.current_chunk_id == 0:
            return
            
        chunk_pnl_pct = ((self.total_capital - self.chunk_start_equity) / self.chunk_start_equity) * 100
        
        self.chunk_pnl[self.current_chunk_id] = {
            'start_equity': self.chunk_start_equity,
            'end_equity': self.total_capital,
            'pnl_pct': chunk_pnl_pct,
            'n_trades': len([t for t in self.trade_history if t.get('chunk_id') == self.current_chunk_id])
        }
        
        logger.info(f"Chunk {self.current_chunk_id} completed with PnL: {chunk_pnl_pct:.2f}% "
                   f"(Equity: ${self.chunk_start_equity:.2f} -> ${self.total_capital:.2f})")
    
    def get_chunk_performance_ratio(self, chunk_id: int, optimal_pnl: float) -> float:
        """
        Calculate the performance ratio for a specific chunk compared to the optimal PnL.
        
        Args:
            chunk_id: The ID of the chunk to calculate the ratio for.
            optimal_pnl: The optimal possible PnL for this chunk.
            
        Returns:
            float: The performance ratio (actual_pnl / optimal_pnl), clipped to [0, 1].
        """
        if chunk_id not in self.chunk_pnl:
            logger.warning(f"No PnL data found for chunk {chunk_id}")
            return 0.0
            
        if optimal_pnl <= 0:
            return 0.0
            
        actual_pnl = self.chunk_pnl[chunk_id]['pnl_pct']
        ratio = actual_pnl / optimal_pnl
        
        # Clip the ratio between 0 and 1 to prevent extreme values
        return max(0.0, min(1.0, ratio))