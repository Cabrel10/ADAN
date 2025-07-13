from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

class FinanceManager:
    def __init__(self, initial_capital: float, fee_pct: float, min_order_usdt: float):
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.min_order_usdt = min_order_usdt
        self.reset()

    def reset(self):
        self.capital_total_usdt = self.initial_capital
        self.capital_libre_usdt = self.initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {} # {'BTCUSDT': {'units': 0.1, 'avg_price': 50000, 'entry_time': datetime.now()}}
        self.peak_capital = self.initial_capital
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.trade_history: List[Dict[str, Any]] = []

    def update_market_value(self, current_prices: Dict[str, float]):
        assets_value = 0.0
        for asset, pos in self.positions.items():
            if asset in current_prices:
                assets_value += pos['units'] * current_prices[asset]
            else:
                # If current price not available, use average entry price as fallback
                assets_value += pos['units'] * pos['avg_price']

        self.capital_total_usdt = self.capital_libre_usdt + assets_value
        self.peak_capital = max(self.peak_capital, self.capital_total_usdt)

    def get_current_drawdown(self) -> float:
        if self.peak_capital == 0: return 0.0
        return (self.peak_capital - self.capital_total_usdt) / self.peak_capital
    
    def can_place_buy_order(self, order_value_usdt: float) -> bool:
        required_capital = order_value_usdt * (1 + self.fee_pct)
        return self.capital_libre_usdt >= required_capital and order_value_usdt >= self.min_order_usdt

    def can_place_sell_order(self, asset: str, units_to_sell: float) -> bool:
        return asset in self.positions and self.positions[asset]['units'] >= units_to_sell and units_to_sell > 0

    def execute_buy(self, asset: str, units: float, price: float) -> bool:
        trade_value = units * price
        fees = trade_value * self.fee_pct
        
        if not self.can_place_buy_order(trade_value):
            return False

        self.capital_libre_usdt -= (trade_value + fees)
        
        current_units = self.positions.get(asset, {}).get('units', 0.0)
        current_avg_price = self.positions.get(asset, {}).get('avg_price', 0.0)

        new_total_value = (current_units * current_avg_price) + trade_value
        new_total_units = current_units + units

        self.positions[asset] = {
            'units': new_total_units,
            'avg_price': new_total_value / new_total_units,
            'entry_time': datetime.now() # Update entry time for simplicity, could be more complex for partial buys
        }
        self.trade_count += 1
        self.trade_history.append({'type': 'buy', 'asset': asset, 'units': units, 'price': price, 'fees': fees, 'timestamp': datetime.now()})
        return True
    
    def execute_sell(self, asset: str, units: float, price: float) -> Optional[float]:
        if not self.can_place_sell_order(asset, units):
            return None # Cannot sell

        position = self.positions[asset]
        trade_value = units * price
        fees = trade_value * self.fee_pct

        self.capital_libre_usdt += (trade_value - fees)

        # Calculate PnL for the sold portion
        pnl = (price - position['avg_price']) * units

        position['units'] -= units
        if position['units'] <= 1e-9: # Effectively zero
            del self.positions[asset]
        
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        else: self.loss_count += 1

        self.trade_history.append({'type': 'sell', 'asset': asset, 'units': units, 'price': price, 'fees': fees, 'pnl': pnl, 'timestamp': datetime.now()})
        return pnl

    def get_performance_metrics(self) -> Dict[str, Any]:
        total_return = ((self.capital_total_usdt - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
        win_rate = (self.win_count / self.trade_count) * 100 if self.trade_count > 0 else 0.0
        
        return {
            'total_capital': self.capital_total_usdt,
            'free_capital': self.capital_libre_usdt,
            'invested_capital': self.capital_total_usdt - self.capital_libre_usdt,
            'current_drawdown': self.get_current_drawdown(),
            'total_return': total_return,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate
        }