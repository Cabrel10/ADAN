#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manages and validates trade orders."""
from ..common.utils import get_logger
from ..portfolio.portfolio_manager import PortfolioManager


logger = get_logger()


class OrderManager:
    def __init__(self, trading_rules: dict, penalties: dict):
        """
        Initializes the OrderManager.
        Args:
            trading_rules: Dictionary of trading rules from the config.
            penalties: Dictionary of penalties for invalid actions.
        """
        self.trading_rules = trading_rules
        self.penalties = penalties
        logger.info("OrderManager initialized.")

    def open_position(
        self, 
        portfolio: PortfolioManager, 
        asset: str, 
        price: float, 
        size: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        confidence: float = 1.0
    ) -> bool:
        """
        Opens a new position via the portfolio manager.

        Args:
            portfolio: The portfolio instance.
            asset: The asset to open a position for.
            price: The current price of the asset.
            size: The size of the position to open. If None, calculated based on risk.
            stop_loss: Optional stop loss price.
            take_profit: Optional take profit price.
            confidence: The confidence level of the action.

        Returns:
            bool: True if the position was opened successfully, False otherwise.
        """
        if portfolio.positions[asset].is_open:
            logger.warning(
                f"Cannot open a new position for {asset}, "
                "one is already open."
            )
            return False

        # If size is not provided, calculate it based on risk
        if size is None:
            stop_loss_pct = self.trading_rules.get('stop_loss', 0.0)
            risk_per_trade = self.trading_rules.get('risk_per_trade', 0.01)
            
            # Get available capital from portfolio
            available_capital = portfolio.get_available_capital()
            
            # Simple position sizing based on risk per trade
            if stop_loss_pct > 0:
                risk_amount = available_capital * risk_per_trade
                size = risk_amount / (stop_loss_pct * price)
            else:
                # Default to 10% of available capital if no stop loss
                size = (available_capital * 0.1) / price
        
        # Round to appropriate decimal places for the asset
        size = round(size, 8)  # 8 decimal places for crypto
        
        if size <= 0:
            logger.warning(
                f"Invalid position size {size} for {asset} "
                f"at price {price}"
            )
            return False
            
        # Open the position through the portfolio manager
        # Note: stop_loss and take_profit are not used in the current implementation
        # of portfolio.open_position, but we keep them in the signature for future use
        return portfolio.open_position(asset, price, size)

    def close_position(
        self, 
        portfolio: PortfolioManager, 
        asset: str, 
        price: float
    ) -> float:
        """
        Closes the current position for a given asset via the portfolio manager.

        Args:
            portfolio: The portfolio instance.
            asset: The asset to close the position for.
            price: The current price of the asset.

        Returns:
            The realized PnL from closing the position.
        """
        if (asset not in portfolio.positions or 
                not portfolio.positions[asset].is_open):
            logger.warning(f"Cannot close a position for {asset}, none is open.")
            return 0.0

        # The portfolio manager handles the logic of closing
        return portfolio.close_position(asset, price)

    def validate_order(
        self, 
        order: dict, 
        portfolio_manager: PortfolioManager
    ) -> tuple[bool, float]:
        """
        Validates a generic trade order.
        Note: This seems to be a legacy method. The primary logic is now in
        open_position and close_position which use the portfolio's own validation.
        """
        # This method's logic is largely incompatible with the current 
        # PortfolioManager structure. It relies on dictionary access to positions 
        # and a 'capital' attribute. The core validation is now handled within 
        # PortfolioManager.validate_position. We can perform a basic check here.
        size = order.get('units', 0)
        price = order.get('price', 0)
        asset = order.get('asset', 'BTC')

        if size == 0 or price <= 0:
            return False, self.penalties.get('invalid_action', 1.0)

        is_valid = portfolio_manager.validate_position(asset, abs(size), price)
        penalty = 0.0 if is_valid else self.penalties.get('invalid_action', 1.0)

        return is_valid, penalty

    def reset(self) -> None:
        """
        Reset the order manager's internal state.
        
        This method is called at the beginning of each episode to reset any 
        internal state. Currently, OrderManager doesn't maintain internal state 
        that needs resetting, but this method is provided for API consistency.
        """
        logger.debug("OrderManager reset called.")
        # No internal state to reset at the moment
