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
from ..common.reward_logger import RewardLogger

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
        
        # Commission and profit threshold parameters
        self.commission_penalty = self.config.get('commission_penalty', 1.5)  # Multiplier for commission penalty
        self.min_profit_multiplier = self.config.get('min_profit_multiplier', 3.0)  # Minimum profit multiple of commission
        
        # Chunk-based reward parameters
        self.optimal_trade_bonus = self.config.get('optimal_trade_bonus', 1.0)
        self.performance_threshold = self.config.get('performance_threshold', 0.8)  # 80% of optimal
        
        # Track chunk information
        self.current_chunk_id = 0
        self.chunk_rewards = {}
        
        # Initialize reward logger
        self.reward_logger = RewardLogger(env_config)
        
        # Episode tracking for detailed logging
        self.current_episode_rewards = []
        self.current_episode_id = 0
        
        # DBE (Dynamic Budgeting Engine) parameters
        self.winrate = 0.5  # Will be updated based on performance
        self.drawdown = 0.0  # Will be updated based on portfolio metrics
        self.risk_level = 0.3  # Initial risk level (0.1 to 1.0)

        logger.info("RewardCalculator initialized with chunk-based rewards and detailed logging.")

    def calculate(self, portfolio_metrics: Dict[str, Any], trade_pnl: float, action: int, 
                 chunk_id: int = None, optimal_chunk_pnl: float = None, performance_ratio: float = None) -> float:
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
            performance_ratio: The performance ratio for the current chunk (actual_pnl / optimal_pnl).

        Returns:
            float: The total reward for the current timestep.
        """
        # Base reward components
        reward = 0.0
        
        # Get commission from portfolio metrics
        commission = portfolio_metrics.get('total_commission', 0.0)
        
        # Apply commission penalty (reduces reward based on commission paid)
        commission_penalty = self.commission_penalty * abs(commission)
        
        # Check if trade meets minimum profit threshold
        if trade_pnl > 0 and commission > 0 and trade_pnl < (self.min_profit_multiplier * commission):
            # Penalize trades that don't meet minimum profit threshold
            reward -= (self.min_profit_multiplier * commission - trade_pnl) * 2
            logger.debug(f"Trade PnL ({trade_pnl:.4f}) below minimum threshold (3x commission = {3*commission:.4f})")
        
        # Reward for realized PnL (when trades are closed)
        reward += (trade_pnl - commission_penalty) * self.pnl_multiplier

        # Small penalty for inaction to encourage trading when opportunities exist
        if action == 0:  # Hold action
            reward += self.inaction_penalty
            
        # Add chunk-based performance bonus if chunk information is provided
        if chunk_id is not None and optimal_chunk_pnl is not None and optimal_chunk_pnl > 0:
            # Only add the bonus once per chunk
            if chunk_id != self.current_chunk_id:
                self.current_chunk_id = chunk_id
                
                # Add bonus if performance exceeds threshold
                if performance_ratio is not None and performance_ratio >= self.performance_threshold:
                    bonus = self.optimal_trade_bonus * (performance_ratio - self.performance_threshold)
                    reward += bonus
                    
                    # Store chunk rewards for analysis
                    self.chunk_rewards[chunk_id] = {
                        'optimal_pnl': optimal_chunk_pnl,
                        'performance_ratio': performance_ratio,
                        'bonus': bonus
                    }
                    
                    logger.info(f"Chunk {chunk_id} performance bonus: {bonus:.4f} "
                              f"(Ratio: {performance_ratio:.2f}, Optimal PnL: {optimal_chunk_pnl:.2f}%)")

        # Incorporate risk metrics into reward (e.g., penalize high drawdown or low Sharpe)
        # This is a placeholder for now, actual implementation will depend on how these metrics are calculated and passed.
        drawdown = portfolio_metrics.get('drawdown', 0.0)
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0.0)

        # Example: Penalize large drawdowns
        if drawdown < -0.05: # If drawdown is worse than -5%
            reward -= abs(drawdown) * 10 # Scale penalty by drawdown magnitude

        # Example: Reward for good Sharpe ratio (if above a certain threshold)
        if sharpe_ratio > 0.5:
            reward += sharpe_ratio * 0.1 # Small bonus for good risk-adjusted returns

        # Clip the reward to prevent extreme values
        reward = np.clip(reward, *self.clipping_range)
        
        # Detailed reward logging
        reward_components = {
            'realized_pnl': trade_pnl * self.pnl_multiplier,
            'inaction_penalty': self.inaction_penalty if action == 0 else 0.0,
            'drawdown_penalty': -abs(drawdown) * 10 if drawdown < -0.05 else 0.0,
            'sharpe_bonus': sharpe_ratio * 0.1 if sharpe_ratio > 0.5 else 0.0,
            'performance_bonus': 0.0  # Will be updated if bonus is applied
        }
        
        # Update performance bonus in components if applicable
        if (chunk_id is not None and chunk_id != self.current_chunk_id and 
            performance_ratio is not None and performance_ratio >= self.performance_threshold):
            bonus = self.optimal_trade_bonus * (performance_ratio - self.performance_threshold)
            reward_components['performance_bonus'] = bonus
            
            # Log performance bonus separately
            self.reward_logger.log_performance_bonus({
                'chunk_id': chunk_id,
                'optimal_pnl': optimal_chunk_pnl,
                'actual_pnl': optimal_chunk_pnl * performance_ratio if optimal_chunk_pnl else 0.0,
                'performance_ratio': performance_ratio,
                'bonus_amount': bonus,
                'threshold': self.performance_threshold
            })
        
        # Update DBE parameters based on performance
        self._update_dbe_parameters(portfolio_metrics)
        
        # Log DBE metrics
        logger.info(
            f"DBE ADAPT | Winrate: {self.winrate:.2%} | "
            f"Drawdown: {self.drawdown:.2%} | "
            f"Risk Level: {self.risk_level:.2f}"
        )
        
        # Log detailed reward calculation
        self.reward_logger.log_reward_calculation({
            'total_reward': reward,
            'components': reward_components,
            'dbe_metrics': {
                'winrate': self.winrate,
                'drawdown': self.drawdown,
                'risk_level': self.risk_level,
                'commission_penalty': commission_penalty
            },
            'metadata': {
                'action': action,
                'trade_pnl': trade_pnl,
                'drawdown': drawdown,
                'sharpe_ratio': sharpe_ratio,
                'chunk_id': chunk_id,
                'performance_ratio': performance_ratio,
                'optimal_chunk_pnl': optimal_chunk_pnl,
                'clipped': bool(reward != np.sum(list(reward_components.values())))
            }
        })
        
        # Track episode rewards
        self.current_episode_rewards.append(reward)

        logger.info(f"Reward calculated: {reward:.4f} (PnL: {trade_pnl:.4f}, Action: {action}, Drawdown: {drawdown:.4f}, Sharpe: {sharpe_ratio:.4f}, Chunk ID: {str(chunk_id)}, Performance Ratio: {str(performance_ratio)})")

        return float(reward)
    
    def _update_dbe_parameters(self, portfolio_metrics: Dict[str, Any]) -> None:
        """
        Update DBE (Dynamic Budgeting Engine) parameters based on portfolio performance.
        
        Args:
            portfolio_metrics: Current portfolio metrics including winrate, drawdown, etc.
        """
        # Update winrate and drawdown from portfolio metrics
        self.winrate = portfolio_metrics.get('win_rate', self.winrate)
        self.drawdown = portfolio_metrics.get('drawdown', self.drawdown)
        
        # Get cash utilization (0.0 to 1.0)
        cash_utilization = portfolio_metrics.get('cash_utilization', 0.0)
        
        # Calculate dynamic risk adjustment
        # Risk increases with higher winrate, lower drawdown, and higher cash utilization
        self.risk_level = max(0.1, min(1.0, 
            self.winrate * (1.0 - min(0.5, self.drawdown)) * (0.5 + cash_utilization * 0.5)
        ))
        
        # Update max position size based on risk level
        if 'max_position_size_pct' in portfolio_metrics:
            portfolio_metrics['max_position_size_pct'] *= self.risk_level
            
        logger.debug(
            f"DBE UPDATE | Winrate: {self.winrate:.2%} | "
            f"Drawdown: {self.drawdown:.2%} | "
            f"Cash Util: {cash_utilization:.2%} | "
            f"New Risk: {self.risk_level:.3f}"
        )

    def finalize_episode(self) -> None:
        """
        Finalise l'épisode actuel et log les métriques d'épisode.
        """
        if not self.current_episode_rewards:
            return
        
        # Calculer les statistiques de l'épisode
        episode_rewards = np.array(self.current_episode_rewards)
        
        episode_data = {
            'total_reward': np.sum(episode_rewards),
            'average_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_length': len(episode_rewards),
            'performance_bonuses': sum(1 for r in self.chunk_rewards.values() if r.get('bonus', 0) > 0),
            'risk_penalties': sum(1 for r in episode_rewards if r < -0.1)
        }
        
        # Logger l'épisode
        self.reward_logger.log_episode_reward(self.current_episode_id, episode_data)
        
        # Réinitialiser pour le prochain épisode
        self.current_episode_rewards = []
        self.current_episode_id += 1
        
        logger.info(f"Episode {self.current_episode_id - 1} finalized: "
                   f"Total reward: {episode_data['total_reward']:.4f}, "
                   f"Length: {episode_data['episode_length']}")
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """
        Obtenir les statistiques détaillées des récompenses.
        
        Returns:
            Dictionnaire contenant les statistiques des récompenses
        """
        return self.reward_logger.get_reward_statistics()
    
    def save_reward_logs(self, filename: str = None) -> None:
        """
        Sauvegarder les logs de récompenses.
        
        Args:
            filename: Nom de fichier optionnel
        """
        self.reward_logger.save_reward_logs(filename)
    
    def generate_reward_report(self) -> str:
        """
        Générer un rapport détaillé des récompenses.
        
        Returns:
            Rapport formaté des récompenses
        """
        return self.reward_logger.generate_reward_report()
