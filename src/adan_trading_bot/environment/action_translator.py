#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Translates agent actions into trade orders."""
import numpy as np
from ..common.utils import get_logger

logger = get_logger()

class ActionTranslator:
    def __init__(self, assets: list[str]):
        self.assets = assets

    def translate_action(self, action: np.ndarray, portfolio_manager, current_prices: dict[str, float]) -> list[dict]:
        """Translates a raw action from the agent into a list of trade orders."""
        orders = []
        tier = portfolio_manager.get_current_tier()
        allocation_per_trade = tier['allocation_per_trade']

        for i, asset in enumerate(self.assets):
            action_value = action[i]
            
            if action_value > 0.5: # Buy signal
                trade_value = portfolio_manager.capital * allocation_per_trade
                units = trade_value / current_prices[asset]
                orders.append({'asset': asset, 'units': units, 'price': current_prices[asset]})
            elif action_value < -0.5: # Sell signal
                units_to_sell = portfolio_manager.positions[asset]['units']
                if units_to_sell > 0:
                    orders.append({'asset': asset, 'units': -units_to_sell, 'price': current_prices[asset]})
        
        return orders
