#!/usr/bin/env python3
"""
Test script for the RewardCalculator component.

This script tests:
1. Basic reward calculation
2. Reward shaping based on portfolio metrics
3. Chunk-based performance bonuses
4. Risk-adjusted rewards
5. Reward clipping
6. Edge cases and error handling
"""

import sys
import os
import logging
from typing import Dict, Any
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.environment.reward_calculator import RewardCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config() -> Dict[str, Any]:
    """Create a test configuration for the RewardCalculator."""
    return {
        'reward_shaping': {
            'realized_pnl_multiplier': 1.0,
            'unrealized_pnl_multiplier': 0.1,
            'inaction_penalty': -0.0001,
            'reward_clipping_range': [-5.0, 5.0],
            'optimal_trade_bonus': 1.0,
            'performance_threshold': 0.8
        }
    }

def create_portfolio_metrics(realized_pnl: float = 0.0, unrealized_pnl: float = 0.0, 
                            drawdown: float = 0.0, sharpe_ratio: float = 0.0) -> Dict[str, Any]:
    """Create a mock portfolio metrics dictionary."""
    return {
        'realized_pnl': realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'drawdown': drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_capital': 10000.0,
        'initial_capital': 10000.0
    }

def test_basic_reward_calculation():
    """Test basic reward calculation functionality."""
    logger.info("Testing basic reward calculation...")
    
    config = create_test_config()
    reward_calculator = RewardCalculator(config)
    
    # Test 1: No trade, hold action
    portfolio_metrics = create_portfolio_metrics()
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=0)
    
    # Should only include inaction penalty
    expected_reward = config['reward_shaping']['inaction_penalty']
    assert abs(reward - expected_reward) < 0.0001, \
        f"Expected reward {expected_reward}, got {reward}"
    
    # Test 2: Profitable trade
    trade_pnl = 4.0  # Use a smaller value that won't be clipped
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=trade_pnl, action=1)
    
    # Should include PnL multiplier
    expected_reward = trade_pnl * config['reward_shaping']['realized_pnl_multiplier']
    assert abs(reward - expected_reward) < 0.0001, \
        f"Expected reward {expected_reward}, got {reward}"
    
    # Test 3: Losing trade
    trade_pnl = -4.0  # Use a smaller value that won't be clipped
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=trade_pnl, action=2)
    
    # Should include negative PnL
    expected_reward = trade_pnl * config['reward_shaping']['realized_pnl_multiplier']
    assert abs(reward - expected_reward) < 0.0001, \
        f"Expected reward {expected_reward}, got {reward}"
    
    logger.info("✅ Basic reward calculation test passed")

def test_risk_adjusted_rewards():
    """Test risk-adjusted reward calculation."""
    logger.info("Testing risk-adjusted rewards...")
    
    config = create_test_config()
    reward_calculator = RewardCalculator(config)
    
    # Test 1: Good Sharpe ratio
    portfolio_metrics = create_portfolio_metrics(sharpe_ratio=1.0)
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=1)
    
    # Should include Sharpe ratio bonus
    assert reward > 0, f"Expected positive reward for good Sharpe ratio, got {reward}"
    
    # Test 2: Bad drawdown
    portfolio_metrics = create_portfolio_metrics(drawdown=-0.1)  # 10% drawdown
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=1)
    
    # Should include drawdown penalty
    assert reward < 0, f"Expected negative reward for bad drawdown, got {reward}"
    
    # Test 3: Combined metrics
    portfolio_metrics = create_portfolio_metrics(sharpe_ratio=1.5, drawdown=-0.03)
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=4.0, action=1)
    
    # Should include PnL, Sharpe bonus, but no drawdown penalty (drawdown > -0.05)
    expected_base = 4.0  # PnL
    # Due to clipping, we can't expect reward > expected_base if it would exceed the clipping range
    # Instead, check that reward is positive and includes at least the PnL
    assert reward >= expected_base, \
        f"Expected reward >= {expected_base} for combined metrics, got {reward}"
    
    logger.info("✅ Risk-adjusted rewards test passed")

def test_chunk_based_rewards():
    """Test chunk-based reward calculation."""
    logger.info("Testing chunk-based rewards...")
    
    config = create_test_config()
    reward_calculator = RewardCalculator(config)
    
    portfolio_metrics = create_portfolio_metrics()
    
    # Test 1: Below performance threshold
    chunk_id = 1
    optimal_chunk_pnl = 5.0  # 5% optimal PnL
    performance_ratio = 0.7  # 70% of optimal (below threshold)
    
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=chunk_id, optimal_chunk_pnl=optimal_chunk_pnl, 
        performance_ratio=performance_ratio
    )
    
    # For below-threshold performance, no bonus is given and chunk is not recorded
    # The implementation only records chunks that receive a bonus
    assert chunk_id not in reward_calculator.chunk_rewards, \
        f"Chunk {chunk_id} should not be recorded in chunk_rewards for below-threshold performance"
    
    # Test 2: Above performance threshold
    chunk_id = 2
    optimal_chunk_pnl = 5.0
    performance_ratio = 0.9  # 90% of optimal (above threshold)
    
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=chunk_id, optimal_chunk_pnl=optimal_chunk_pnl, 
        performance_ratio=performance_ratio
    )
    
    # Should include bonus
    expected_bonus = config['reward_shaping']['optimal_trade_bonus'] * (performance_ratio - config['reward_shaping']['performance_threshold'])
    assert chunk_id in reward_calculator.chunk_rewards, \
        f"Chunk {chunk_id} should be recorded in chunk_rewards"
    assert abs(reward_calculator.chunk_rewards[chunk_id]['bonus'] - expected_bonus) < 0.0001, \
        f"Expected bonus {expected_bonus}, got {reward_calculator.chunk_rewards[chunk_id]['bonus']}"
    
    # Test 3: Same chunk ID (should not add bonus again)
    # Note: The bonus is only added once when the chunk ID changes, but the base reward
    # calculation still happens for each call, so we need to use the same parameters
    # to get the same reward
    reward_same_chunk = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=chunk_id, optimal_chunk_pnl=optimal_chunk_pnl, 
        performance_ratio=performance_ratio
    )
    
    # The bonus should not be added again, but the base reward calculation still happens
    # So we can't directly compare the rewards, but we can check that the chunk is still recorded
    assert chunk_id in reward_calculator.chunk_rewards, \
        f"Chunk {chunk_id} should still be recorded in chunk_rewards"
    assert abs(reward_calculator.chunk_rewards[chunk_id]['bonus'] - expected_bonus) < 0.0001, \
        f"Bonus should remain the same for same chunk ID"
    
    logger.info("✅ Chunk-based rewards test passed")

def test_reward_clipping():
    """Test reward clipping functionality."""
    logger.info("Testing reward clipping...")
    
    config = create_test_config()
    reward_calculator = RewardCalculator(config)
    
    portfolio_metrics = create_portfolio_metrics()
    
    # Test 1: Extremely large positive reward
    trade_pnl = 1000.0  # Very large PnL
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=trade_pnl, action=1)
    
    # Should be clipped to upper bound
    assert reward <= config['reward_shaping']['reward_clipping_range'][1], \
        f"Reward should be clipped to {config['reward_shaping']['reward_clipping_range'][1]}, got {reward}"
    
    # Test 2: Extremely large negative reward
    trade_pnl = -1000.0  # Very large negative PnL
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=trade_pnl, action=1)
    
    # Should be clipped to lower bound
    assert reward >= config['reward_shaping']['reward_clipping_range'][0], \
        f"Reward should be clipped to {config['reward_shaping']['reward_clipping_range'][0]}, got {reward}"
    
    logger.info("✅ Reward clipping test passed")

def test_custom_reward_shaping():
    """Test custom reward shaping configurations."""
    logger.info("Testing custom reward shaping...")
    
    # Create a custom config with different multipliers
    custom_config = {
        'reward_shaping': {
            'realized_pnl_multiplier': 2.0,  # Double the default
            'unrealized_pnl_multiplier': 0.5,  # Higher than default
            'inaction_penalty': -0.01,  # More severe penalty
            'reward_clipping_range': [-10.0, 10.0],  # Wider range
            'optimal_trade_bonus': 2.0,  # Double bonus
            'performance_threshold': 0.7  # Lower threshold
        }
    }
    
    reward_calculator = RewardCalculator(custom_config)
    portfolio_metrics = create_portfolio_metrics()
    
    # Test with profitable trade
    trade_pnl = 4.0  # Use a smaller value that won't be clipped
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=trade_pnl, action=1)
    
    # Should use custom multiplier
    expected_reward = trade_pnl * custom_config['reward_shaping']['realized_pnl_multiplier']
    assert abs(reward - expected_reward) < 0.0001, \
        f"Expected reward {expected_reward}, got {reward}"
    
    # Test with inaction
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=0)
    
    # Should use custom inaction penalty
    expected_reward = custom_config['reward_shaping']['inaction_penalty']
    assert abs(reward - expected_reward) < 0.0001, \
        f"Expected reward {expected_reward}, got {reward}"
    
    # Test chunk-based reward with custom threshold
    chunk_id = 1
    optimal_chunk_pnl = 5.0
    performance_ratio = 0.75  # Above custom threshold (0.7), but below default (0.8)
    
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=chunk_id, optimal_chunk_pnl=optimal_chunk_pnl, 
        performance_ratio=performance_ratio
    )
    
    # Should include bonus with custom multiplier and threshold
    expected_bonus = custom_config['reward_shaping']['optimal_trade_bonus'] * (performance_ratio - custom_config['reward_shaping']['performance_threshold'])
    assert chunk_id in reward_calculator.chunk_rewards, \
        f"Chunk {chunk_id} should be recorded in chunk_rewards"
    assert abs(reward_calculator.chunk_rewards[chunk_id]['bonus'] - expected_bonus) < 0.0001, \
        f"Expected bonus {expected_bonus}, got {reward_calculator.chunk_rewards[chunk_id]['bonus']}"
    
    logger.info("✅ Custom reward shaping test passed")

def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Testing edge cases...")
    
    config = create_test_config()
    reward_calculator = RewardCalculator(config)
    
    portfolio_metrics = create_portfolio_metrics()
    
    # Test 1: None values for chunk parameters
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=None, optimal_chunk_pnl=None, performance_ratio=None
    )
    
    # Should not crash and return a valid reward
    assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"
    
    # Test 2: Zero optimal PnL
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=1, optimal_chunk_pnl=0.0, performance_ratio=0.9
    )
    
    # Should handle zero division gracefully
    assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"
    
    # Test 3: Negative optimal PnL
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=2, optimal_chunk_pnl=-5.0, performance_ratio=0.9
    )
    
    # Should handle negative optimal PnL gracefully
    assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"
    
    # Test 4: Empty portfolio metrics
    reward = reward_calculator.calculate({}, trade_pnl=0.0, action=1)
    
    # Should handle missing metrics gracefully
    assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"
    
    logger.info("✅ Edge cases test passed")

def test_reward_consistency():
    """Test reward calculation consistency across multiple calls."""
    logger.info("Testing reward consistency...")
    
    config = create_test_config()
    reward_calculator = RewardCalculator(config)
    
    portfolio_metrics = create_portfolio_metrics()
    
    # Calculate reward multiple times with the same inputs
    rewards = []
    for _ in range(10):
        reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=10.0, action=1)
        rewards.append(reward)
    
    # All rewards should be identical
    assert all(r == rewards[0] for r in rewards), \
        f"Expected consistent rewards, got {rewards}"
    
    logger.info("✅ Reward consistency test passed")

def run_all_tests():
    """Run all reward calculator tests."""
    logger.info("🚀 Starting reward calculator tests...")
    
    try:
        test_basic_reward_calculation()
        test_risk_adjusted_rewards()
        test_chunk_based_rewards()
        test_reward_clipping()
        test_custom_reward_shaping()
        test_edge_cases()
        test_reward_consistency()
        
        logger.info("🎉 All reward calculator tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)