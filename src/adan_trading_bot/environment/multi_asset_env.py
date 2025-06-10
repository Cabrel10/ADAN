import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich import box
import os
import time

from ..common.utils import get_logger, ensure_dir_exists
from ..common.constants import HOLD, BUY, SELL, ACTION_HOLD, ACTION_BUY_ASSET_0, ACTION_SELL_ASSET_0, ORDER_TYPE_MARKET
from .state_builder import StateBuilder
from .order_manager import OrderManager
from .reward_calculator import RewardCalculator

logger = get_logger()
console = Console()

class MultiAssetEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df_received, config, encoder=None, max_episode_steps_override=None):
        super(MultiAssetEnv, self).__init__()
        self.config = config
        self.assets = sorted(list(set(self.config.get('data', {}).get('assets', []))))
        if not self.assets: raise ValueError("No assets defined in config.")
        
        if df_received is None or df_received.empty:
            raise ValueError("MultiAssetEnv received an empty or None DataFrame.")
        self.df = df_received.copy()
        
        self.encoder = encoder
        env_config = config.get('environment', {})
        self.initial_capital = env_config.get('initial_capital', 10000.0)
        self.transaction_config = env_config.get('transaction', {})
        self.order_rules_config = env_config.get('order_rules', {})
        self.penalties_config = env_config.get('penalties', {})
        self.fee_percent = self.transaction_config.get('fee_percent', 0.001)
        self.fixed_fee = self.transaction_config.get('fixed_fee', 0.0)
        self.min_order_value_tolerable = self.order_rules_config.get('min_value_tolerable', 10.0)
        self.min_order_value_absolute = self.order_rules_config.get('min_value_absolute', 9.0)

        if len(self.assets) > 5: self.assets = self.assets[:5]
        
        data_config = config.get('data', {})
        self.cnn_input_window_size = data_config.get('cnn_input_window_size', 20)
        self.training_timeframe = data_config.get('training_timeframe', '1h')
        
        current_base_features_per_asset = []
        if self.training_timeframe == '1m':
            base_1m_features = data_config.get('base_market_features', ['open', 'high', 'low', 'close', 'volume'])
            current_base_features_per_asset.extend(base_1m_features)
        else: # For '1h' or '1d'
            current_base_features_per_asset.extend(['open', 'high', 'low', 'close', 'volume'])
            # MODIFIED LOGIC FOR BASE FEATURE NAMES (NO SUFFIX HERE)
            indicators_config_for_timeframe = data_config.get('indicators_by_timeframe', {}).get(self.training_timeframe, [])
            for indicator_spec in indicators_config_for_timeframe:
                indicator_base_name = indicator_spec.get('output_col_name') # Prefer output_col_name
                if not indicator_base_name and isinstance(indicator_spec.get('output_col_names'), list) and indicator_spec['output_col_names']:
                    indicator_base_name = indicator_spec['output_col_names'][0] # Fallback to first of output_col_names
                if not indicator_base_name: indicator_base_name = indicator_spec.get('alias') # Fallback
                if not indicator_base_name: indicator_base_name = indicator_spec.get('name') # Fallback

                if indicator_base_name:
                    current_base_features_per_asset.append(indicator_base_name) # APPEND BASE NAME
                else:
                    logger.warning(f"Could not determine base name for indicator spec: {indicator_spec} for tf {self.training_timeframe}")

        self.base_feature_names = list(dict.fromkeys(current_base_features_per_asset))
        logger.info(f"MultiAssetEnv: Final base_feature_names for {self.training_timeframe}: {self.base_feature_names}")
        
        self.num_market_features_per_step = len(self.base_feature_names) * len(self.assets)
        self.num_input_channels = 1
        self.image_shape = (self.num_input_channels, self.cnn_input_window_size, self.num_market_features_per_step)
        
        self.state_builder = StateBuilder(config, self.assets, encoder=encoder,
                                         base_feature_names=self.base_feature_names, 
                                         cnn_input_window_size=self.cnn_input_window_size)
        self.order_manager = OrderManager(config)
        self.reward_calculator = RewardCalculator(config)
        
        self.action_space = spaces.Discrete(1 + 2 * len(self.assets))
        obs_space_dims = self.state_builder.get_observation_space_dim()
        self.image_shape = obs_space_dims["image_features"]
        self.num_market_features_per_step = self.image_shape[2]

        self.observation_space = spaces.Dict({
            "image_features": spaces.Box(low=-np.inf, high=np.inf, shape=self.image_shape, dtype=np.float32),
            "vector_features": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_dims["vector_features"],), dtype=np.float32)
        })
        
        self.max_steps = len(self.df)
        if max_episode_steps_override is not None: self.max_steps = min(max_episode_steps_override, self.max_steps)
        self.export_history = env_config.get('export_history', True)
        self.export_dir = env_config.get('export_dir', None)
        logger.info(f"MultiAssetEnv initialized with {len(self.assets)} assets and {self.max_steps} steps")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.capital = self.initial_capital
        self.positions = {}
        self.current_step = 0
        self.history = []
        self.trade_log = []
        self.cumulative_reward = 0.0
        self.order_manager.clear_pending_orders()
        observation = self._get_observation()
        info = {"portfolio_value": self.initial_capital, "capital": self.capital, "positions": self.positions.copy(), "step": self.current_step}
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        old_portfolio_value = self._calculate_portfolio_value()
        current_prices = self._get_current_prices()
        
        reward_mod, executed_orders, self.capital, self.positions = self.order_manager.process_pending_orders(
            current_prices, self.capital, self.positions, self.current_step)
        for order_info in executed_orders: self.trade_log.append({"step": self.current_step, "type": "EXECUTED_ORDER", **order_info})

        asset_id, action_type = self._translate_action(action)
        trade_info = {}
        if action_type != HOLD:
            if asset_id in current_prices and not np.isnan(current_prices[asset_id]): # Check for NaN price
                current_price = current_prices[asset_id]
                current_tier = self.reward_calculator.get_current_tier(self.capital)
                if action_type == BUY:
                    if len(self.positions) >= current_tier["max_positions"] and asset_id not in self.positions:
                        penalty = self.penalties_config.get('max_positions_reached', -0.2)
                        reward_mod_action, status, trade_info = penalty, "INVALID_MAX_POSITIONS", {"reason": f"Max positions ({current_tier['max_positions']}) reached", "reward_mod": penalty}
                    else:
                        allocated_value_usdt = self._get_position_size(asset_id, current_price, current_tier)
                        reward_mod_action, status, trade_info = self.order_manager.execute_order(
                            asset_id, action_type, current_price, self.capital, self.positions,
                            allocated_value_usdt=allocated_value_usdt, order_type=ORDER_TYPE_MARKET, current_step=self.current_step)
                        if status == "BUY_EXECUTED": self.capital -= trade_info["total_cost"]
                else: # SELL
                    reward_mod_action, status, trade_info = self.order_manager.execute_order(
                        asset_id, action_type, current_price, self.capital, self.positions,
                        order_type=ORDER_TYPE_MARKET, current_step=self.current_step)
                    if status == "SELL_EXECUTED": self.capital += trade_info["value"] - trade_info["fee"]
                reward_mod += reward_mod_action
                self.trade_log.append({"step": self.current_step, "action": action, "action_type": action_type, "asset_id": asset_id, "status": status, "reward_mod": reward_mod_action, **trade_info})
            else:
                penalty = self.penalties_config.get('price_not_available', -0.1)
                reward_mod += penalty
                trade_info = {"reason": f"Price not available or NaN for {asset_id}", "reward_mod": penalty, "status": "PRICE_UNAVAILABLE"}
        else: # HOLD
            reward_mod += self.penalties_config.get('time_step', -0.001)
            trade_info = {"reason": "HOLD action"}

        new_portfolio_value = self._calculate_portfolio_value()
        current_tier = self.reward_calculator.get_current_tier(self.capital) # Recalculate tier after action
        reward = self.reward_calculator.calculate_reward(old_portfolio_value, new_portfolio_value, penalties=-reward_mod, tier=current_tier)
        self.cumulative_reward += reward
        
        observation = self._get_observation()
        done = (self.capital <= self.min_order_value_absolute and not self.positions) or (self.current_step >= self.max_steps - 1)
        
        self.history.append({"step": self.current_step, "action": action, "reward": reward, "cumulative_reward": self.cumulative_reward, "portfolio_value": new_portfolio_value, "capital": self.capital, "positions": self.positions.copy(), "old_portfolio_value": old_portfolio_value, "reward_mod": reward_mod, "done": done})
        # self._display_trading_table(action, old_portfolio_value, new_portfolio_value, reward, reward_mod, trade_info) # Commented out for brevity
        info = {"portfolio_value": new_portfolio_value, "capital": self.capital, "positions": self.positions.copy(), "step": self.current_step, "reward_mod": reward_mod, "trade_info": trade_info, "tier": current_tier}
        self.current_step += 1
        return observation, reward, done, False, info

    def _get_observation(self):
        market_data_window = self._get_market_data_window()
        if market_data_window.empty:
            logger.warning(f"Empty market data window at step {self.current_step}. Using zeros for observation.")
            image_features = np.zeros(self.image_shape, dtype=np.float32)
            vector_features = np.zeros(1 + len(self.assets), dtype=np.float32)
            vector_features[0] = 1.0
            return {"image_features": image_features, "vector_features": vector_features}
        
        observation = self.state_builder.build_observation(
            market_data_window=market_data_window, capital=self.capital, positions=self.positions,
            image_shape=self.image_shape) # Removed apply_scaling, StateBuilder handles it
        return observation
        
    def _get_market_data_window(self):
        start_idx = max(0, self.current_step - self.cnn_input_window_size + 1)
        end_idx = self.current_step + 1
        window = self.df.iloc[start_idx:end_idx].copy()
        return window
    
    def _get_current_data_row(self):
        return self.df.iloc[self.current_step]
    
    def _get_current_prices(self):
        current_prices = {}
        data_row = self._get_current_data_row()
        for asset in self.assets:
            close_col = f'close_{asset}' # This needs to be present in data_row
            if close_col in data_row:
                current_prices[asset] = data_row[close_col]
            else:
                logger.error(f"Price column {close_col} not found for asset {asset} at step {self.current_step}. Available: {list(data_row.index[:10])}")
                current_prices[asset] = np.nan # Important to return NaN if price missing
        return current_prices
    
    def _translate_action(self, action):
        if action == ACTION_HOLD: return None, HOLD
        num_assets = len(self.assets)
        if ACTION_BUY_ASSET_0 <= action < ACTION_BUY_ASSET_0 + num_assets:
            return self.assets[action - ACTION_BUY_ASSET_0], BUY
        if ACTION_SELL_ASSET_0 <= action < ACTION_SELL_ASSET_0 + num_assets:
            return self.assets[action - ACTION_SELL_ASSET_0], SELL
        return None, HOLD
    
    def _calculate_portfolio_value(self):
        total_value = self.capital
        current_prices = self._get_current_prices()
        for asset, position in self.positions.items():
            if asset in current_prices and not np.isnan(current_prices[asset]):
                total_value += position['qty'] * current_prices[asset]
        return total_value
    
    def _get_position_size(self, asset_id, price, current_tier):
        allocation_frac = current_tier.get('allocation_frac_per_pos', 0.95)
        return self.capital * allocation_frac

    def render(self, mode='human'): pass
    def close(self):
        if self.export_history and self.history: self.export_trading_data(self.export_dir)
    def export_trading_data(self, export_dir=None): pass # Simplified for brevity
    def _calculate_performance_metrics(self): return {} # Simplified
