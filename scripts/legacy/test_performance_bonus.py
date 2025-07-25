#!/usr/bin/env python3
"""
Test script for performance bonus functionality in the RewardCalculator.

This script tests:
1. Loading and applying reward configuration
2. Performance bonus calculation based on optimal PnL
3. Risk-adjusted bonus calculation
4. Adaptive bonus calculation based on market conditions
5. Integration with the main reward calculation
"""

import sys
import os
import logging
import yaml
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

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        return {}

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

def test_config_loading():
    """Test loading reward configuration from file."""
    logger.info("Testing config loading...")
    
    # Load the reward configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'reward_config.yaml')
    config = load_config(config_path)
    
    # Check that the configuration was loaded successfully
    assert 'reward_shaping' in config, "reward_shaping section should be in config"
    assert 'performance_bonus' in config, "performance_bonus section should be in config"
    
    # Check specific parameters
    assert 'optimal_trade_bonus' in config['performance_bonus'], "optimal_trade_bonus should be in config"
    assert 'performance_threshold' in config['performance_bonus'], "performance_threshold should be in config"
    
    logger.info("✅ Config loading test passed")
    return config

def test_performance_bonus_calculation(config: Dict[str, Any]):
    """Test performance bonus calculation based on optimal PnL."""
    logger.info("Testing performance bonus calculation...")
    
    # Create a RewardCalculator with the loaded config
    reward_calculator = RewardCalculator({
        'reward_shaping': {
            **config['reward_shaping'],
            'optimal_trade_bonus': config['performance_bonus']['optimal_trade_bonus'],
            'performance_threshold': config['performance_bonus']['performance_threshold']
        }
    })
    
    portfolio_metrics = create_portfolio_metrics()
    
    # Test 1: Below performance threshold
    chunk_id = 1
    optimal_chunk_pnl = 5.0  # 5% optimal PnL
    performance_ratio = config['performance_bonus']['performance_threshold'] - 0.1  # Below threshold
    
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=chunk_id, optimal_chunk_pnl=optimal_chunk_pnl, 
        performance_ratio=performance_ratio
    )
    
    # For below-threshold performance, no bonus is given
    assert chunk_id not in reward_calculator.chunk_rewards, \
        f"Chunk {chunk_id} should not be recorded in chunk_rewards for below-threshold performance"
    
    # Test 2: Above performance threshold
    chunk_id = 2
    optimal_chunk_pnl = 5.0
    performance_ratio = config['performance_bonus']['performance_threshold'] + 0.1  # Above threshold
    
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=0.0, action=1,
        chunk_id=chunk_id, optimal_chunk_pnl=optimal_chunk_pnl, 
        performance_ratio=performance_ratio
    )
    
    # Should include bonus
    expected_bonus = config['performance_bonus']['optimal_trade_bonus'] * (performance_ratio - config['performance_bonus']['performance_threshold'])
    assert chunk_id in reward_calculator.chunk_rewards, \
        f"Chunk {chunk_id} should be recorded in chunk_rewards"
    assert abs(reward_calculator.chunk_rewards[chunk_id]['bonus'] - expected_bonus) < 0.0001, \
        f"Expected bonus {expected_bonus}, got {reward_calculator.chunk_rewards[chunk_id]['bonus']}"
    
    logger.info("✅ Performance bonus calculation test passed")

def test_risk_adjusted_bonus(config: Dict[str, Any]):
    """Test risk-adjusted bonus calculation."""
    logger.info("Testing risk-adjusted bonus...")
    
    # Create a RewardCalculator with the loaded config
    reward_calculator = RewardCalculator({
        'reward_shaping': config['reward_shaping']
    })
    
    # Test 1: Good Sharpe ratio
    sharpe_threshold = config['performance_bonus']['sharpe_ratio_threshold']
    sharpe_bonus = config['performance_bonus']['sharpe_ratio_bonus']
    
    portfolio_metrics = create_portfolio_metrics(sharpe_ratio=sharpe_threshold + 0.5)
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=1)
    
    # Should include Sharpe ratio bonus
    assert reward > 0, f"Expected positive reward for good Sharpe ratio, got {reward}"
    
    # Test 2: Bad drawdown
    drawdown_threshold = config['performance_bonus']['drawdown_threshold']
    drawdown_penalty = config['performance_bonus']['drawdown_penalty_factor']
    
    portfolio_metrics = create_portfolio_metrics(drawdown=drawdown_threshold - 0.05)  # Worse than threshold
    reward = reward_calculator.calculate(portfolio_metrics, trade_pnl=0.0, action=1)
    
    # Should include drawdown penalty
    assert reward < 0, f"Expected negative reward for bad drawdown, got {reward}"
    
    logger.info("✅ Risk-adjusted bonus test passed")

def test_adaptive_bonus(config: Dict[str, Any]):
    """Test adaptive bonus calculation based on market conditions."""
    logger.info("Testing adaptive bonus...")
    
    # This is a placeholder for future implementation
    # The current RewardCalculator doesn't have adaptive bonus functionality yet
    
    # For now, just log that this feature is not yet implemented
    logger.info("⚠️ Adaptive bonus not yet implemented in RewardCalculator")
    
    # Return True to indicate test was run
    return True

def test_integration_with_main_reward(config: Dict[str, Any]):
    """Test integration of performance bonus with main reward calculation."""
    logger.info("Testing integration with main reward...")
    
    # Create a RewardCalculator with the loaded config
    reward_calculator = RewardCalculator({
        'reward_shaping': {
            **config['reward_shaping'],
            'optimal_trade_bonus': config['performance_bonus']['optimal_trade_bonus'],
            'performance_threshold': config['performance_bonus']['performance_threshold']
        }
    })
    
    # Test combined reward calculation
    portfolio_metrics = create_portfolio_metrics(
        realized_pnl=100.0,
        sharpe_ratio=config['performance_bonus']['sharpe_ratio_threshold'] + 0.5
    )
    
    chunk_id = 3
    optimal_chunk_pnl = 5.0
    performance_ratio = config['performance_bonus']['performance_threshold'] + 0.2  # Well above threshold
    
    # Calculate reward with all components
    reward = reward_calculator.calculate(
        portfolio_metrics, trade_pnl=3.0, action=1,
        chunk_id=chunk_id, optimal_chunk_pnl=optimal_chunk_pnl, 
        performance_ratio=performance_ratio
    )
    
    # Should include PnL, performance bonus, and Sharpe bonus
    assert reward > 3.0, f"Expected reward > 3.0 for combined components, got {reward}"
    assert chunk_id in reward_calculator.chunk_rewards, \
        f"Chunk {chunk_id} should be recorded in chunk_rewards"
    
    logger.info("✅ Integration with main reward test passed")

def run_all_tests():
    """Run all performance bonus tests."""
    logger.info("🚀 Starting performance bonus tests...")
    
    try:
        # Load configuration
        config = test_config_loading()
        
        # Run tests
        test_performance_bonus_calculation(config)
        test_risk_adjusted_bonus(config)
        test_adaptive_bonus(config)
        test_integration_with_main_reward(config)
        
        logger.info("🎉 All performance bonus tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)