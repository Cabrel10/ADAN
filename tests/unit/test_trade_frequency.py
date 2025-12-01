#!/usr/bin/env python3
"""
Unit Tests for TradeFrequencyController
Tests all critical functionality including the deadlock fix.
"""

import unittest
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from adan_trading_bot.environment.trade_frequency_controller import (
    TradeFrequencyController,
    FrequencyConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestTradeFrequency")


class TestTradeFrequencyController(unittest.TestCase):
    """Comprehensive unit tests for TradeFrequencyController."""
    
    def setUp(self):
        """Initialize controller with test configuration."""
        self.config = FrequencyConfig(
            min_interval_steps=6,
            daily_trade_limit=10,
            asset_cooldown_steps=3,
            force_trade_steps_by_tf={
                "5m": 15,
                "1h": 20,
                "4h": 50
            }
        )
        self.controller = TradeFrequencyController(self.config)
    
    def test_initialization(self):
        """Test controller initializes with correct defaults."""
        self.assertEqual(self.controller.daily_trade_count, 0)
        self.assertEqual(self.controller.last_trade_step, -999)
        self.assertEqual(self.controller.current_day, 0)
        logger.info("✅ Initialization test passed")
    
    def test_record_trade_natural(self):
        """Test that natural trades increment daily count."""
        initial_count = self.controller.daily_trade_count
        
        self.controller.record_trade(
            asset="BTCUSDT",
            current_step=10,
            timeframe="5m",
            is_forced=False
        )
        
        self.assertEqual(
            self.controller.daily_trade_count,
            initial_count + 1,
            "Natural trade should increment daily_trade_count"
        )
        logger.info("✅ Natural trade recording test passed")
    
    def test_record_trade_forced(self):
        """Test that forced trades DO NOT increment daily count (DEADLOCK FIX)."""
        initial_count = self.controller.daily_trade_count
        
        self.controller.record_trade(
            asset="BTCUSDT",
            current_step=10,
            timeframe="5m",
            is_forced=True
        )
        
        self.assertEqual(
            self.controller.daily_trade_count,
            initial_count,
            "Forced trade should NOT increment daily_trade_count"
        )
        logger.info("✅ Forced trade recording test passed (deadlock fix verified)")
    
    def test_multiple_force_trades_do_not_block(self):
        """
        Critical regression test: Multiple force trades should not block natural trades.
        This is the exact deadlock scenario that was fixed.
        """
        # Execute 10 force trades (fill the daily limit in force trades)
        for i in range(10):
            self.controller.record_trade(
                asset="BTCUSDT",
                current_step=i * 10,
                is_forced=True
            )
        
        # Daily count should still be 0
        self.assertEqual(self.controller.daily_trade_count, 0)
        
        # Natural trade should still be allowed
        can_trade, reason = self.controller.can_open_trade(
            asset="ETHUSDT",
            current_step=100
        )
        
        self.assertTrue(can_trade, f"Natural trade blocked after force trades: {reason}")
        logger.info("✅ Force trades deadlock regression test passed")
    
    def test_daily_limit_enforcement(self):
        """Test that daily limit correctly blocks trades."""
        # Fill the daily limit with natural trades
        for i in range(self.config.daily_trade_limit):
            self.controller.record_trade(
                asset=f"ASSET{i}",
                current_step=i * 10,
                is_forced=False
            )
        
        # Verify count
        self.assertEqual(self.controller.daily_trade_count, self.config.daily_trade_limit)
        
        # Next trade should be blocked
        can_trade, reason = self.controller.can_open_trade(
            asset="BTCUSDT",
            current_step=1000,
            check_daily=True
        )
        
        self.assertFalse(can_trade, "Trade should be blocked when daily limit reached")
        self.assertIn("Daily limit reached", reason)
        logger.info("✅ Daily limit enforcement test passed")
    
    def test_global_interval_enforcement(self):
        """Test minimum interval between any trades."""
        # Record a trade
        self.controller.record_trade(
            asset="BTCUSDT",
            current_step=100,
            is_forced=False
        )
        
        # Try to trade too soon (within min_interval_steps)
        can_trade, reason = self.controller.can_open_trade(
            asset="ETHUSDT",
            current_step=103,  # Only 3 steps later, need 6
            check_global=True
        )
        
        self.assertFalse(can_trade)
        self.assertIn("Global interval not met", reason)
        
        # After sufficient interval, should be allowed
        can_trade, reason = self.controller.can_open_trade(
            asset="ETHUSDT",
            current_step=106,  # 6 steps later
            check_global=True
        )
        
        self.assertTrue(can_trade)
        logger.info("✅ Global interval enforcement test passed")
    
    def test_asset_cooldown_enforcement(self):
        """Test per-asset cooldown period."""
        # Trade BTCUSDT
        self.controller.record_trade(
            asset="BTCUSDT",
            current_step=100,
            is_forced=False
        )
        
        # Try to trade same asset too soon
        can_trade, reason = self.controller.can_open_trade(
            asset="BTCUSDT",
            current_step=101,  # Only 1 step later, need 3
            check_asset=True,
            check_global=False,  # Disable global check for this test
            check_daily=False
        )
        
        self.assertFalse(can_trade)
        self.assertIn("Asset cooldown active", reason)
        
        # After cooldown, should be allowed
        can_trade, reason = self.controller.can_open_trade(
            asset="BTCUSDT",
            current_step=103,  # 3 steps later
            check_asset=True,
            check_global=False,
            check_daily=False
        )
        
        self.assertTrue(can_trade)
        logger.info("✅ Asset cooldown enforcement test passed")
    
    def test_daily_reset(self):
        """Test that daily reset clears counters."""
        # Record some trades
        for i in range(5):
            self.controller.record_trade(
                asset="BTCUSDT",
                current_step=i * 10,
                is_forced=False
            )
        
        self.assertEqual(self.controller.daily_trade_count, 5)
        
        # Reset to new day
        self.controller.reset_daily(new_day=1)
        
        self.assertEqual(self.controller.daily_trade_count, 0)
        self.assertEqual(len(self.controller.asset_trade_count), 0)
        logger.info("✅ Daily reset test passed")
    
    def test_should_force_trade(self):
        """Test force trade threshold logic."""
        # Record a trade first to establish a known state
        self.controller.record_trade("BTCUSDT", 100, "5m", False)
        
        # Check soon after - should NOT force (within threshold)
        should_force = self.controller.should_force_trade(
            current_step=110,  # Only 10 steps since last trade at 100
            timeframe="5m"  # threshold is 15
        )
        self.assertFalse(should_force, "Should not force trade within threshold")
        
        # After threshold, SHOULD force
        should_force = self.controller.should_force_trade(
            current_step=116,  # 16 steps since last trade
            timeframe="5m"
        )
        self.assertTrue(should_force, "Should force trade after threshold exceeded")
        logger.info("✅ Force trade threshold test passed")
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        self.controller.record_trade(
            asset="BTCUSDT",
            current_step=100,
            timeframe="5m",
            is_forced=False
        )
        
        stats = self.controller.get_stats()
        
        self.assertIn('daily_trade_count', stats)
        self.assertIn('daily_limit', stats)
        self.assertIn('last_trade_step', stats)
        self.assertEqual(stats['daily_trade_count'], 1)
        self.assertEqual(stats['last_trade_step'], 100)
        logger.info("✅ Statistics retrieval test passed")


if __name__ == '__main__':
    unittest.main()
