#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from src.adan_trading_bot.environment.reward_calculator import RewardCalculator

class TestRewardCalculator(unittest.TestCase):
    def setUp(self):
        self.reward_config = {
            'log_return_multiplier': 1.0,
            'clipping_range': [-1.0, 1.0]
        }
        self.calculator = RewardCalculator(self.reward_config)

    def test_positive_reward(self):
        tier = {'reward_multiplier': 1.0}
        reward = self.calculator.calculate_reward(100, 110, 0, tier)
        self.assertGreater(reward, 0)

    def test_negative_reward(self):
        tier = {'reward_multiplier': 1.0}
        reward = self.calculator.calculate_reward(100, 90, 0, tier)
        self.assertLess(reward, 0)

    def test_penalty(self):
        tier = {'reward_multiplier': 1.0}
        reward = self.calculator.calculate_reward(100, 100, 0.5, tier)
        self.assertLess(reward, 0)

    def test_clipping(self):
        tier = {'reward_multiplier': 1.0}
        reward = self.calculator.calculate_reward(100, 1000, 0, tier) # Should be clipped
        self.assertEqual(reward, self.reward_config['clipping_range'][1])

if __name__ == '__main__':
    unittest.main()
