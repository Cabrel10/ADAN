#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Realistic Trading Environment for ADAN 2.0.
Inherits from MultiAssetChunkedEnv and integrates:
1. TradeFrequencyController (Cooldown, Daily Limits)
2. StableRewardCalculator (Normalized rewards)
3. Circuit Breaker (Risk management)
4. Min Notional (Binance limits)
5. Candle Sync (Live mode alignment)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .multi_asset_chunked_env import MultiAssetChunkedEnv
from .trade_frequency_controller import TradeFrequencyController, FrequencyConfig
from .stable_reward_calculator import StableRewardCalculator
from .market_friction import (
    AdaptiveSlippage,
    LatencySimulator,
    LiquidityModel,
    BinanceFeeModel,
    MarketConditions
)


class RealisticTradingEnv(MultiAssetChunkedEnv):
    """
    Enhanced environment with modular constraints for realistic trading.
    Per ADAN 2.0 Spec Phase 2.
    """
    
    def __init__(
        self,
        live_mode: bool = False,
        min_hold_steps: int = 6,  # 30 mins (6 * 5m)
        cooldown_steps: int = 3,  # 15 mins (3 * 5m)
        min_notional: float = 10.0,  # $10 USDT
        circuit_breaker_pct: float = 0.15,  # 15% max drawdown
        daily_trade_limit: int = 10,
        use_stable_reward: bool = True,  # Toggle for new reward calculator
        enable_market_friction: bool = True,  # Toggle for friction models
        reward_config: Optional[Dict[str, float]] = None,
        friction_config: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.live_mode = live_mode
        self.min_hold_steps = min_hold_steps
        self.min_notional = min_notional
        self.circuit_breaker_pct = circuit_breaker_pct
        self.use_stable_reward = use_stable_reward
        
        # Initialize TradeFrequencyController
        freq_config = FrequencyConfig(
            min_interval_steps=1,  # Allow 1 step between trades globally
            daily_trade_limit=daily_trade_limit,
            asset_cooldown_steps=cooldown_steps,
            force_trade_steps_by_tf=self.config.get("trading_rules", {}).get("force_trade_steps_by_timeframe", {
                "5m": 15,
                "1h": 20,
                "4h": 50
            })
        )
        self.freq_controller = TradeFrequencyController(freq_config)
        
        # Initialize StableRewardCalculator
        if self.use_stable_reward:
            r_cfg = reward_config or {}
            self.reward_calculator = StableRewardCalculator(
                pnl_normalization_factor=r_cfg.get("pnl_normalization", 100.0),
                sharpe_weight=r_cfg.get("sharpe_weight", 0.2),
                drawdown_penalty_weight=r_cfg.get("drawdown_weight", 0.3),
                frequency_penalty_weight=r_cfg.get("frequency_weight", 0.1),
                consistency_bonus_weight=r_cfg.get("consistency_weight", 0.1)
            )
        
        # Circuit breaker state
        self.circuit_breaker_triggered = False
        
        # Hold minimum tracking (per-asset entry steps)
        self.asset_entry_steps: Dict[str, int] = {}
        
        # Initialize Market Friction Models
        self.enable_market_friction = enable_market_friction
        if self.enable_market_friction:
            f_cfg = friction_config or {}
            self.slippage_model = AdaptiveSlippage(
                base_slippage_bps=f_cfg.get("slippage_bps", 2.0),
                size_impact_factor=0.1,
                volatility_impact_factor=0.5
            )
            self.latency_model = LatencySimulator(
                min_latency_ms=50.0,
                max_latency_ms=200.0,
                price_drift_per_ms=0.00001
            )
            self.liquidity_model = LiquidityModel(
                depth_factor=0.001,
                impact_exponent=1.5
            )
            self.fee_model = BinanceFeeModel(
                maker_fee=f_cfg.get("fee_bps", 0.04) / 100.0,
                taker_fee=f_cfg.get("fee_bps", 0.04) / 100.0,
                tier="VIP0",
                use_bnb_discount=False
            )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🛡️ RealisticTradingEnv initialized (Live={live_mode}, StableReward={use_stable_reward}, Friction={enable_market_friction})")
        self.logger.info(f"   Hold={min_hold_steps} steps, Cooldown={cooldown_steps} steps, DailyLimit={daily_trade_limit}")
        self.logger.info(f"   Min Notional=${min_notional}, Circuit Breaker={circuit_breaker_pct:.1%}")

    def reset(self, **kwargs):
        """Reset environment and all controllers."""
        obs = super().reset(**kwargs)
        
        # Reset controllers
        self.freq_controller.reset()
        if self.use_stable_reward:
            self.reward_calculator.reset()
        
        # Reset state tracking
        self.circuit_breaker_triggered = False
        self.asset_entry_steps.clear()
        
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Override step to enforce global constraints and use StableRewardCalculator.
        """
        # 1. Circuit Breaker Check
        if self.circuit_breaker_triggered:
            self.logger.warning("🚨 CIRCUIT BREAKER ACTIVE - Trading halted.")
            return self._get_observation(), -1.0, True, False, {"circuit_breaker": True}
            
        # Calculate current drawdown
        initial = self.portfolio_manager.initial_equity
        current = self.portfolio_manager.current_value
        if initial > 0:
            current_pnl_pct = (current - initial) / initial
        else:
            current_pnl_pct = 0.0

        if current_pnl_pct <= -self.circuit_breaker_pct:
            self.logger.critical(
                f"🚨 CIRCUIT BREAKER TRIGGERED! Drawdown {current_pnl_pct:.2%} > {self.circuit_breaker_pct:.2%}"
            )
            self.circuit_breaker_triggered = True
            # Attempt to close all positions
            try:
                if hasattr(self.portfolio_manager, 'close_all_positions'):
                    self.portfolio_manager.close_all_positions(self._get_current_prices())
            except Exception as e:
                self.logger.error(f"Failed to close positions on circuit breaker: {e}")
            return self._get_observation(), -10.0, True, False, {"circuit_breaker": True}

        # 2. Candle Sync (Live Mode Only)
        if self.live_mode:
            self._sync_to_candle()

        # 3. Execute parent step (will call _execute_trades internally)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 4. Override reward with StableRewardCalculator if enabled
        if self.use_stable_reward:
            # Calculate step PnL from info or estimate
            step_pnl = info.get('pnl', 0.0)
            trade_count = info.get('trades_executed', 0)
            
            reward_breakdown = self.reward_calculator.calculate_reward(
                pnl=step_pnl,
                portfolio_value=current,
                initial_value=initial,
                trade_count=trade_count
            )
            reward = reward_breakdown['total_reward']
            info['reward_breakdown'] = reward_breakdown

        return obs, reward, terminated, truncated, info

    def _sync_to_candle(self):
        """
        Placeholder for candle synchronization in live mode.
        Real implementation should be handled by the runner/orchestrator.
        """
        pass

    def _execute_trades(
        self,
        action: np.ndarray,
        dbe_modulation: dict,
        action_threshold: float,
        force_trade: bool = False,
    ) -> tuple[float, int]:
        """
        Override _execute_trades to integrate TradeFrequencyController and enforce constraints.
        """
        if not hasattr(self, "portfolio_manager"):
            return 0.0, 0, 0

        current_prices = self._get_current_prices()
        if not current_prices:
            return 0.0, 0, 0

        # Update positions first (mark to market)
        pnl_from_update, sl_tp_receipts = self.portfolio_manager.update_market_price(
            current_prices, self.current_step
        )
        if sl_tp_receipts:
            self._step_closed_receipts.extend(sl_tp_receipts)
        
        realized_pnl = pnl_from_update
        first_discrete_action = 0
        trades_executed_this_step = 0

        # Iterate assets and apply frequency/hold constraints
        for i, asset in enumerate(self.assets):
            if i * 3 + 2 >= len(action) or asset not in current_prices:
                continue

            base_idx = i * 3
            main_decision = action[base_idx + 0]
            
            # 🔍 DIAGNOSTIC LOGGING - See what model produces
            if i == 0 and self.current_step % 100 == 0:  # Log only for first asset, every 100 steps
                self.logger.info(
                    f"[ACTION_DIAG] Step {self.current_step} | "
                    f"Sample action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] | "
                    f"Stats: min={action.min():.3f}, max={action.max():.3f}, mean={action.mean():.3f}"
                )
            
            # --- FREQUENCY CONTROLLER CHECKS ---
            
            # 1. Check if we can open a trade for this asset
            can_trade, reason = self.freq_controller.can_open_trade(
                asset=asset,
                current_step=self.current_step,
                check_global=True,
                check_asset=True,
                check_daily=True
            )
            
            if not can_trade and abs(main_decision) > action_threshold:
                # Block trade if frequency constraints violated
                if self.current_step % 500 == 0:  # Log occasionally for diagnosis
                    self.logger.debug(f"❄️ Trade blocked for {asset}: {reason}")
                action[base_idx + 0] = 0.0  # Neutralize action
                continue
            
            # 2. Check hold minimum (prevent flickering)
            position = self.portfolio_manager.positions.get(asset)
            if position and position.is_open:
                entry_step = self.asset_entry_steps.get(asset, -9999)
                steps_held = self.current_step - entry_step
                
                if steps_held < self.min_hold_steps:
                    # Block CLOSING if hold minimum not met
                    if main_decision < -action_threshold:  # Trying to sell/close
                        if self.current_step % 500 == 0:
                            self.logger.debug(f"⏳ Hold Min for {asset}: {steps_held}/{self.min_hold_steps}")
                        action[base_idx + 0] = 0.0
                        continue
            
            # 3. Record trade if action is significant (for tracking)
            if abs(action[base_idx + 0]) > action_threshold:
                # Record opening a position
                if not (position and position.is_open):
                    self.asset_entry_steps[asset] = self.current_step
                    self.freq_controller.record_trade(
                        asset=asset,
                        current_step=self.current_step,
                        timeframe=getattr(self, 'current_timeframe_for_trade', '5m'),
                        is_forced=False  # Natural trade
                    )
                    trades_executed_this_step += 1

        # Call parent implementation with filtered actions
        # Parent will call OrderManager which handles Min Notional checks
        result = super()._execute_trades(action, dbe_modulation, action_threshold, force_trade)
        
        # Store trades count for reward calculation
        if hasattr(self, '_step_info'):
            self._step_info['trades_executed'] = trades_executed_this_step
        
        return result

    def apply_market_friction(
        self,
        target_price: float,
        order_size_usd: float,
        side: str = "buy",
        asset: str = "BTCUSDT"
    ) -> Dict[str, float]:
        """
        Apply all market friction models to determine final execution price.
        
        Args:
            target_price: Intended execution price
            order_size_usd: Order size in USDT
            side: "buy" or "sell"
            asset: Asset symbol
            
        Returns:
            Dictionary with breakdown of execution costs
        """
        if not self.enable_market_friction:
            # No friction - return target price with basic fee
            basic_fee = order_size_usd * 0.001  # 0.1%
            return {
                'target_price': target_price,
                'slippage': 0.0,
                'latency_impact': 0.0,
                'liquidity_impact': 0.0,
                'execution_price': target_price,
                'fee': basic_fee,
                'total_cost': order_size_usd + (basic_fee if side == "buy" else -basic_fee)
            }
        
        # Estimate market conditions (simplified)
        market_conditions = MarketConditions(
            volatility=0.01,  # Default 1%
            spread_bps=5.0,
            volume_24h=1e9
        )
        
        # Apply friction models sequentially
        price_after_slippage = self.slippage_model.apply_slippage(
            target_price, order_size_usd, market_conditions, side
        )
        slippage = price_after_slippage - target_price
        
        price_after_latency = self.latency_model.apply_latency(
            price_after_slippage, market_conditions
        )
        latency_impact = price_after_latency - price_after_slippage
        
        final_price = self.liquidity_model.apply_impact(
            price_after_latency, order_size_usd, market_conditions, side
        )
        liquidity_impact = final_price - price_after_latency
        
        # Calculate fee
        fee = self.fee_model.calculate_fee(order_size_usd, is_maker=False)
        
        return {
            'target_price': float(target_price),
            'slippage': float(slippage),
            'latency_impact': float(latency_impact),
            'liquidity_impact': float(liquidity_impact),
            'execution_price': float(final_price),
            'fee': float(fee),
            'total_cost': float((order_size_usd / target_price) * final_price + (fee if side == "buy" else -fee))
        }
