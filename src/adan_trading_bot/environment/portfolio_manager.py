#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manages the portfolio, including capital, positions, and PnL."""
import numpy as np
import pandas as pd
from ..common.utils import get_logger

logger = get_logger()

class PortfolioManager:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PortfolioManager.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.initial_capital = config['portfolio']['initial_capital']
        self.assets = config['assets'] # Assuming assets are defined in the main config
        self.fee_percent = config['trading']['fee_percent']
        self.capital_tiers = config['risk_management']['capital_tiers']
        self.reset()

    def reset(self):
        """Resets the portfolio to its initial state."""
        self.capital = self.initial_capital
        self.positions = {asset: {'units': 0.0, 'avg_price': 0.0} for asset in self.assets}
        self.history = []
        self.update_portfolio_value(None)

    def update_portfolio_value(self, current_prices: dict[str, float] | None):
        """Calculates the total value of the portfolio."""
        market_value = 0.0
        if current_prices:
            for asset, position in self.positions.items():
                market_value += position['units'] * current_prices[asset]
        
        self.portfolio_value = self.capital + market_value
        self.history.append(self.portfolio_value)

    def get_current_tier(self) -> dict:
        """Determines the current capital tier based on portfolio value."""
        for tier in reversed(self.capital_tiers):
            if self.portfolio_value >= tier['threshold']:
                return tier
        return self.capital_tiers[0]

    def get_total_position_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculates the total market value of all open positions.
        
        Args:
            current_prices: Dictionary of current prices for all assets.

        Returns:
            The total market value of open positions.
        """
        total_value = 0.0
        for asset, position in self.positions.items():
            if position['units'] != 0:
                total_value += position['units'] * current_prices.get(asset, 0.0) # Use current price
        return total_value

    def update_dbe_params(self, dbe_modulation: Dict[str, Any]) -> None:
        """
        Updates portfolio manager parameters based on DBE modulation.
        Currently, this is a placeholder for future dynamic adjustments.
        """
        # Example: if DBE suggests a new max position size, update it here
        # self.max_position_size = dbe_modulation.get('position_size_pct', self.max_position_size)
        pass

    def execute_trade(self, asset: str, units: float, price: float):
        """
        Executes a trade and updates the portfolio.
        
        Args:
            asset: The asset to trade (e.g., 'BTC').
            units: Number of units to trade. Positive for buy, negative for sell.
            price: The price per unit of the asset.
        """
        trade_value = units * price
        fees = abs(trade_value) * self.fee_percent
        
        self.capital -= fees  # Deduct transaction fees from the capital

        if units > 0:  # Buy
            self.capital -= trade_value
            current_units = self.positions[asset]['units']
            current_avg_price = self.positions[asset]['avg_price']
            new_total_value = (current_units * current_avg_price) + trade_value
            new_total_units = current_units + units
            self.positions[asset]['units'] = new_total_units
            self.positions[asset]['avg_price'] = new_total_value / new_total_units
        elif units < 0:  # Sell
            self.capital += abs(trade_value)
            self.positions[asset]['units'] += units
            if self.positions[asset]['units'] == 0:
                self.positions[asset]['avg_price'] = 0.0
                
    def update_market_price(self, price: float):
        """
        Updates the current market price for the asset and recalculates portfolio value.
        
        Args:
            price: The current market price of the asset.
        """
        # For a single asset portfolio, we can update the portfolio value directly
        current_prices = {asset: price for asset in self.positions.keys()}
        self.update_portfolio_value(current_prices)
        
    def get_state_features(self) -> np.ndarray:
        """
        Returns the portfolio state features as a numpy array.
        
        Returns:
            A numpy array containing portfolio state features:
            - Current cash balance (normalized by initial capital)
            - Current position size (in units)
            - Current position value (normalized by portfolio value)
            - Current PnL (unrealized + realized, normalized)
        """
        # Get the first (and only) asset position
        asset = next(iter(self.positions))
        position = self.positions[asset]
        
        # Calculate position value
        position_value = position['units'] * position['avg_price'] if position['units'] != 0 else 0.0
        
        # Calculate PnL (simplified for now)
        pnl = (self.portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0.0
        
        # Create feature vector
        features = np.array([
            self.capital / self.initial_capital,  # Normalized cash
            position['units'],                    # Current position size
            position_value / self.portfolio_value if self.portfolio_value > 0 else 0.0,  # Position value ratio
            pnl                                   # Normalized PnL
        ], dtype=np.float32)
        
        return features
        
    def get_metrics(self) -> dict:
        """
        Returns a dictionary of portfolio metrics for logging and monitoring.
        
        Returns:
            A dictionary containing various portfolio metrics.
        """
        # Get the first (and only) asset position
        asset = next(iter(self.positions))
        position = self.positions[asset]
        
        # Calculate position value and PnL
        position_value = position['units'] * position['avg_price'] if position['units'] != 0 else 0.0
        total_pnl = self.portfolio_value - self.initial_capital
        pnl_pct = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
        
        return {
            'capital': self.capital,
            'portfolio_value': self.portfolio_value,
            'initial_capital': self.initial_capital,
            'position_units': position['units'],
            'position_avg_price': position['avg_price'],
            'position_value': position_value,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'is_position_open': position['units'] != 0
        }
