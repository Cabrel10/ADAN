#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validates trade orders based on portfolio state and environment rules."""
from ..common.utils import get_logger

logger = get_logger()

class OrderManager:
    def __init__(self, trading_rules: dict, penalties: dict):
        self.min_order_value = trading_rules['min_order_value']
        self.penalties = penalties

    def open_position(self, portfolio, price: float) -> float:
        """
        Opens a new long position.
        
        Args:
            portfolio: The portfolio instance.
            price: The current price of the asset.
            
        Returns:
            The PnL from the trade.
        """
        # Calculate position size based on portfolio value and risk parameters
        position_size = portfolio.capital * 0.1  # Example: 10% of capital per trade
        units = position_size / price
        
        # Execute the trade
        portfolio.execute_trade('BTC', units, price)  # Assuming 'BTC' as the asset
        logger.info(f"Opened new position: {units:.6f} units at {price:.2f}")
        
        # Return 0 PnL as this is an opening trade
        return 0.0
        
    def close_position(self, portfolio, price: float) -> float:
        """
        Closes the current position.
        
        Args:
            portfolio: The portfolio instance.
            price: The current price of the asset.
            
        Returns:
            The PnL from closing the position.
        """
        # Get current position
        position = portfolio.positions.get('BTC', {'units': 0, 'avg_price': 0})
        if position['units'] == 0:
            return 0.0
            
        # Calculate PnL
        pnl = (price - position['avg_price']) * position['units']
        
        # Close the position
        portfolio.execute_trade('BTC', -position['units'], price)
        logger.info(f"Closed position: {position['units']:.6f} units at {price:.2f} (PnL: {pnl:.2f})")
        
        return pnl

    def validate_order(self, order: dict, portfolio_manager) -> tuple[bool, float]:
        """Validates a trade order.

        Args:
            order: The order to validate.
            portfolio_manager: The portfolio manager instance.

        Returns:
            A tuple containing a boolean indicating if the order is valid and a penalty value.
        """
        asset = order.get('asset', 'BTC')  # Default to 'BTC' if not specified
        units = order['units']
        price = order['price']
        trade_value = abs(units * price)

        # Check minimum order value
        if trade_value < self.min_order_value:
            logger.warning(f"Invalid order: trade value {trade_value:.2f} is below minimum {self.min_order_value:.2f}")
            return False, self.penalties['invalid_action']

        # Check for sufficient funds (for buy orders)
        if units > 0 and portfolio_manager.capital < trade_value:
            logger.warning(f"Invalid order: insufficient funds to buy {units:.6f} of {asset}")
            return False, self.penalties['insufficient_funds']

        # Check if trying to sell more than owned
        position = portfolio_manager.positions.get(asset, {'units': 0})
        if units < 0 and abs(units) > position['units']:
            logger.warning(f"Invalid order: trying to sell {abs(units):.6f} {asset} but only {position['units']:.6f} owned")
            return False, self.penalties['position_not_found']

        # Check if max positions reached (for new positions)
        tier = portfolio_manager.get_current_tier()
        if units > 0 and position['units'] == 0:
            current_positions = sum(1 for pos in portfolio_manager.positions.values() if pos['units'] > 0)
            if current_positions >= tier['max_positions']:
                logger.warning(f"Invalid order: max positions of {tier['max_positions']} reached")
                return False, self.penalties['max_positions_reached']

        return True, 0.0