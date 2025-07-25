#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import logging
from adan_trading_bot.environment.reward_calculator import RewardCalculator

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

class TestRewardCalculator(unittest.TestCase):
    def setUp(self):
        self.reward_config = {
            'realized_pnl_multiplier': 1.0,
            'unrealized_pnl_multiplier': 0.1,
            'inaction_penalty': -0.0001,
            'reward_clipping_range': [-5.0, 5.0],
            'optimal_trade_bonus': 1.0,
            'performance_threshold': 0.8
        }
        self.calculator = RewardCalculator(self.reward_config)

    def test_realized_pnl_reward(self):
        portfolio_metrics = {'drawdown': 0.0, 'sharpe_ratio': 0.0} # Set Sharpe to 0 to avoid bonus
        reward = self.calculator.calculate(portfolio_metrics, trade_pnl=10.0, action=1) # Buy action
        self.assertAlmostEqual(reward, 5.0) # Should be clipped to 5.0

    def test_inaction_penalty(self):
        portfolio_metrics = {'drawdown': 0.0, 'sharpe_ratio': 0.0} # Set Sharpe to 0 to avoid bonus
        reward = self.calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=0) # Hold action
        self.assertAlmostEqual(reward, self.reward_config['inaction_penalty'])

    def test_chunk_performance_bonus(self):
        portfolio_metrics = {'drawdown': 0.0, 'sharpe_ratio': 0.0} # Set Sharpe to 0 to avoid bonus
        # Simulate a new chunk with high performance
        reward = self.calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=0, 
                                           chunk_id=1, optimal_chunk_pnl=100.0, performance_ratio=0.9)
        expected_bonus = self.reward_config['optimal_trade_bonus'] * (0.9 - self.reward_config['performance_threshold'])
        self.assertAlmostEqual(reward, self.reward_config['inaction_penalty'] + expected_bonus)

    def test_drawdown_penalty(self):
        portfolio_metrics = {'drawdown': -0.1, 'sharpe_ratio': 0.0} # 10% drawdown, Set Sharpe to 0
        reward = self.calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=0)
        expected_penalty = -abs(-0.1) * 10 # From the reward calculator logic
        self.assertAlmostEqual(reward, self.reward_config['inaction_penalty'] + expected_penalty)

    def test_sharpe_ratio_bonus(self):
        portfolio_metrics = {'drawdown': 0.0, 'sharpe_ratio': 0.6} # Good Sharpe, Set drawdown to 0
        reward = self.calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=0)
        expected_bonus = 0.6 * 0.1 # From the reward calculator logic
        self.assertAlmostEqual(reward, self.reward_config['inaction_penalty'] + expected_bonus)

    def test_reward_clipping(self):
        portfolio_metrics = {'drawdown': 0.0, 'sharpe_ratio': 1.0}
        # Large PnL to trigger clipping
        reward = self.calculator.calculate(portfolio_metrics, trade_pnl=1000.0, action=1)
        self.assertEqual(reward, self.reward_config['reward_clipping_range'][1])

        # Large negative PnL to trigger clipping
        reward = self.calculator.calculate(portfolio_metrics, trade_pnl=-1000.0, action=1)
        self.assertEqual(reward, self.reward_config['reward_clipping_range'][0])

if __name__ == '__main__':
    unittest.main()
