#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reward calculation module for the ADAN trading bot.

This module defines the logic for calculating the reward signal that guides the
reinforcement learning agent.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RewardCalculator:
    """
    Calculates the reward for a given step in the trading environment.

    The reward function is designed to guide the agent towards profitable and
    consistent trading behavior.
    """
    def __init__(self, env_config: Dict[str, Any]):
        """
        Initializes the RewardCalculator.

        Args:
            env_config: The environment configuration dictionary, containing the
                        `reward_shaping` section.
        """
        self.config = env_config.get('reward_shaping', {})
        self.pnl_multiplier = self.config.get('realized_pnl_multiplier', 1.0)
        self.unrealized_pnl_multiplier = self.config.get('unrealized_pnl_multiplier', 0.1)
        self.inaction_penalty = self.config.get('inaction_penalty', -0.0001)
        self.clipping_range = self.config.get('reward_clipping_range', [-5.0, 5.0])
        
        # Chunk-based reward parameters
        self.optimal_trade_bonus = self.config.get('optimal_trade_bonus', 1.0)
        self.performance_threshold = self.config.get('performance_threshold', 0.8)  # 80% of optimal
        
        # Track chunk information
        self.current_chunk_id = 0
        self.chunk_rewards = {}

        logger.info("RewardCalculator initialized with chunk-based rewards.")

    def calculate(self, portfolio_metrics: Dict[str, Any], trade_pnl: float, action: int, 
                 chunk_id: int = None, optimal_chunk_pnl: float = None) -> float:
        """
        Calculates the total reward for the current timestep.

        Args:
            portfolio_metrics: A dictionary of performance metrics from the
                               PortfolioManager.
            trade_pnl: The realized profit or loss from a trade executed in the
                       current step. Zero if no trade was closed.
            action: The action taken by the agent (0: Hold, 1: Buy, 2: Sell).
            chunk_id: The current chunk ID for chunk-based rewards.
            optimal_chunk_pnl: The optimal possible PnL for the current chunk.

        Returns:
            float: The total reward for the current timestep.
        """
        # Base reward components
        reward = 0.0

        # Reward for realized PnL (when trades are closed)
        reward += trade_pnl * self.pnl_multiplier

        # Small penalty for inaction to encourage trading when opportunities exist
        if action == 0:  # Hold action
            reward += self.inaction_penalty
            
        # Add chunk-based performance bonus if chunk information is provided
        if chunk_id is not None and optimal_chunk_pnl is not None and optimal_chunk_pnl > 0:
            # Only add the bonus once per chunk
            if chunk_id != self.current_chunk_id:
                self.current_chunk_id = chunk_id
                
                # Get the performance ratio for this chunk
                if hasattr(portfolio_metrics, 'get_chunk_performance_ratio'):
                    performance_ratio = portfolio_metrics.get_chunk_performance_ratio(
                        chunk_id, optimal_chunk_pnl
                    )
                    
                    # Add bonus if performance exceeds threshold
                    if performance_ratio >= self.performance_threshold:
                        bonus = self.optimal_trade_bonus * performance_ratio
                        reward += bonus
                        
                        # Store chunk rewards for analysis
                        self.chunk_rewards[chunk_id] = {
                            'optimal_pnl': optimal_chunk_pnl,
                            'performance_ratio': performance_ratio,
                            'bonus': bonus
                        }
                        
                        logger.info(f"Chunk {chunk_id} performance bonus: {bonus:.4f} "
                                  f"(Ratio: {performance_ratio:.2f}, Optimal PnL: {optimal_chunk_pnl:.2f}%)")

        # Clip the reward to prevent extreme values
        reward = np.clip(reward, *self.clipping_range)

        return float(reward)
