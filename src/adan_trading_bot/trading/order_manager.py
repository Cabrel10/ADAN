#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Order management module for the ADAN trading bot.

In a backtesting environment, this module translates agent actions into simple
portfolio operations without interacting with a live exchange.
"""

import logging
from enum import Enum
from ..portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"

class Order:
    def __init__(self, id: str, symbol: str, type: OrderType, side: OrderSide, price: float, quantity: float, status: OrderStatus = OrderStatus.NEW):
        self.id = id
        self.symbol = symbol
        self.type = type
        self.side = side
        self.price = price
        self.quantity = quantity
        self.status = status

class OrderManager:
    """
    Manages order execution within the backtesting environment.

    This class acts as a simplified broker, translating the agent's discrete
    actions (Hold, Buy, Sell) into state changes in the PortfolioManager.
    """
    def __init__(self, portfolio_manager: PortfolioManager):
        """
        Initializes the OrderManager.

        Args:
            portfolio_manager: An instance of the PortfolioManager to interact with.
        """
        self.portfolio = portfolio_manager
        logger.info("OrderManager initialized for backtesting.")

    def execute_action(self, action: int, current_price: float) -> float:
        """
        Executes a trading action based on the agent's decision.

        Args:
            action: The discrete action from the agent.
                    - 0: Hold (do nothing)
                    - 1: Buy (open a long position)
                    - 2: Sell (close the long position)
            current_price: The current market price for the asset.

        Returns:
            The realized PnL from the action, if any. Returns 0.0 for
            hold or buy actions, or if a sell action is invalid.
        """
        realized_pnl = 0.0

        if action == 1:  # Buy Action
            # Attempt to open a new position
            # This is an "all-in" strategy for simplicity
            if not self.portfolio.position.is_open:
                self.portfolio.open_position(current_price)
                logger.debug(f"Action: BUY at {current_price:.2f}")
            else:
                # If a position is already open, buying again is treated as holding
                logger.debug("Action: Attempted BUY, but position already open. Holding.")

        elif action == 2:  # Sell Action
            # Attempt to close the existing position
            if self.portfolio.position.is_open:
                realized_pnl = self.portfolio.close_position(current_price)
                logger.debug(f"Action: SELL at {current_price:.2f}, PnL: {realized_pnl:.2f}")
            else:
                # If no position is open, selling is treated as holding
                logger.debug("Action: Attempted SELL, but no position open. Holding.")
        
        # For action == 0 (Hold), we do nothing.
        
        return realized_pnl