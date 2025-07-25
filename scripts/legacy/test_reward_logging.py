#!/usr/bin/env python3
"""
Test script for the RewardLogger component.

This script tests:
1. Basic reward logging functionality
2. Performance bonus logging
3. Episode reward tracking
4. Reward statistics and analysis
5. Saving and loading reward logs
6. Report generation
"""

import sys
import os
import logging
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.common.reward_logger import RewardLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config(temp_dir: str) -> Dict[str, Any]:
    """Create a test configuration for the RewardLogger."""
    return {
        'reward_logging': {
            'enabled': True,
            'log_level': 'INFO',
            'save_interval': 5,  # Save every 5 rewards for testing
            'max_history': 100,
            'base_path': temp_dir
        }
    }

def test_basic_reward_logging():
    """Test basic reward logging functionality."""
    logger.info("Testing basic reward logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        reward_logger = RewardLogger(config)
        
        # Test reward calculation logging
        reward_data = {
            'total_reward': 0.5,
            'components': {
                'pnl_reward': 0.3,
                'risk_penalty': -0.1,
                'inaction_penalty': -0.001,
                'bonus': 0.301
            },
            'metadata': {
                'action': 1,
                'episode': 1,
                'step': 10
            }
        }
        
        reward_logger.log_reward_calculation(reward_data)
        
        # Verify reward was logged
        assert len(reward_logger.reward_history) == 1
        assert reward_logger.total_rewards_logged == 1
        
        logged_reward = reward_logger.reward_history[0]
        assert logged_reward['total_reward'] == 0.5
        assert 'pnl_reward' in logged_reward['components']
        
        # Test component history
        assert 'pnl_reward' in reward_logger.component_history
        assert len(reward_logger.component_history['pnl_reward']) == 1
        
    logger.info("✅ Basic reward logging test passed")

def test_performance_bonus_logging():
    """Test performance bonus logging."""
    logger.info("Testing performance bonus logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        reward_logger = RewardLogger(config)
        
        # Test bonus logging
        bonus_data = {
            'chunk_id': 1,
            'optimal_pnl': 5.0,
            'actual_pnl': 4.0,
            'performance_ratio': 0.8,
            'bonus_amount': 0.2,
            'threshold': 0.7
        }
        
        reward_logger.log_performance_bonus(bonus_data)
        
        # Verify bonus was logged
        assert len(reward_logger.bonus_history) == 1
        
        logged_bonus = reward_logger.bonus_history[0]
        assert logged_bonus['chunk_id'] == 1
        assert logged_bonus['bonus_amount'] == 0.2
        assert logged_bonus['performance_ratio'] == 0.8
        
    logger.info("✅ Performance bonus logging test passed")

def test_episode_reward_tracking():
    """Test episode reward tracking."""
    logger.info("Testing episode reward tracking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        reward_logger = RewardLogger(config)
        
        # Test episode logging
        episode_data = {
            'total_reward': 10.5,
            'average_reward': 0.21,
            'reward_std': 0.15,
            'min_reward': -0.5,
            'max_reward': 1.2,
            'episode_length': 50,
            'performance_bonuses': 3,
            'risk_penalties': 2
        }
        
        reward_logger.log_episode_reward(1, episode_data)
        
        # Verify episode was logged
        assert 1 in reward_logger.episode_rewards
        
        logged_episode = reward_logger.episode_rewards[1]
        assert logged_episode['total_reward'] == 10.5
        assert logged_episode['episode_length'] == 50
        assert logged_episode['performance_bonuses'] == 3
        
    logger.info("✅ Episode reward tracking test passed")

def test_reward_statistics():
    """Test reward statistics calculation."""
    logger.info("Testing reward statistics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        reward_logger = RewardLogger(config)
        
        # Add multiple rewards
        rewards = [0.1, 0.2, -0.1, 0.3, 0.0, 0.4, -0.2, 0.5]
        for i, reward in enumerate(rewards):
            reward_data = {
                'total_reward': reward,
                'components': {'base': reward},
                'metadata': {'step': i}
            }
            reward_logger.log_reward_calculation(reward_data)
        
        # Get statistics
        stats = reward_logger.get_reward_statistics()
        
        # Verify statistics
        assert 'mean_reward' in stats
        assert 'std_reward' in stats
        assert 'total_rewards_logged' in stats
        assert stats['total_rewards_logged'] == len(rewards)
        
        # Check mean calculation
        expected_mean = sum(rewards) / len(rewards)
        assert abs(stats['mean_reward'] - expected_mean) < 0.001
        
        # Check percentiles
        assert 'reward_percentiles' in stats
        assert '25th' in stats['reward_percentiles']
        assert '75th' in stats['reward_percentiles']
        
    logger.info("✅ Reward statistics test passed")

def test_save_load_logs():
    """Test saving and loading reward logs."""
    logger.info("Testing save/load logs...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        reward_logger = RewardLogger(config)
        
        # Add some data
        reward_data = {
            'total_reward': 0.5,
            'components': {'base': 0.5},
            'metadata': {'test': True}
        }
        reward_logger.log_reward_calculation(reward_data)
        
        bonus_data = {
            'chunk_id': 1,
            'bonus_amount': 0.1,
            'performance_ratio': 0.8
        }
        reward_logger.log_performance_bonus(bonus_data)
        
        # Save logs
        filename = 'test_reward_logs.json'
        reward_logger.save_reward_logs(filename)
        
        # Verify file was created
        filepath = Path(temp_dir) / filename
        assert filepath.exists()
        
        # Load logs into new logger
        reward_logger2 = RewardLogger(config)
        success = reward_logger2.load_reward_logs(filename)
        
        assert success
        assert len(reward_logger2.reward_history) > 0
        assert len(reward_logger2.bonus_history) > 0
        
    logger.info("✅ Save/load logs test passed")

def test_report_generation():
    """Test reward report generation."""
    logger.info("Testing report generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        reward_logger = RewardLogger(config)
        
        # Add some data
        for i in range(10):
            reward_data = {
                'total_reward': 0.1 * i,
                'components': {
                    'base': 0.05 * i,
                    'bonus': 0.05 * i
                },
                'metadata': {'step': i}
            }
            reward_logger.log_reward_calculation(reward_data)
        
        # Generate report
        report = reward_logger.generate_reward_report()
        
        # Verify report content
        assert isinstance(report, str)
        assert 'REWARD SYSTEM REPORT' in report
        assert 'Mean Reward:' in report
        assert 'COMPONENT CONTRIBUTIONS:' in report
        
        # Test empty report
        empty_logger = RewardLogger(config)
        empty_report = empty_logger.generate_reward_report()
        assert 'No reward data available' in empty_report
        
    logger.info("✅ Report generation test passed")

def test_trend_analysis():
    """Test reward trend analysis."""
    logger.info("Testing trend analysis...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        reward_logger = RewardLogger(config)
        
        # Add increasing rewards
        for i in range(20):
            reward_data = {
                'total_reward': 0.01 * i,  # Increasing trend
                'components': {'base': 0.01 * i},
                'metadata': {'step': i}
            }
            reward_logger.log_reward_calculation(reward_data)
        
        # Test trend calculation
        rewards = [0.01 * i for i in range(20)]
        trend = reward_logger._calculate_trend(rewards)
        assert trend == 'up'
        
        # Test decreasing trend
        decreasing_rewards = [1.0 - 0.05 * i for i in range(20)]
        trend_down = reward_logger._calculate_trend(decreasing_rewards)
        assert trend_down == 'down'
        
        # Test stable trend
        stable_rewards = [0.5] * 20
        trend_stable = reward_logger._calculate_trend(stable_rewards)
        assert trend_stable == 'stable'
        
    logger.info("✅ Trend analysis test passed")

def test_disabled_logging():
    """Test behavior when logging is disabled."""
    logger.info("Testing disabled logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        config['reward_logging']['enabled'] = False
        reward_logger = RewardLogger(config)
        
        # Try to log rewards
        reward_data = {
            'total_reward': 0.5,
            'components': {'base': 0.5},
            'metadata': {}
        }
        reward_logger.log_reward_calculation(reward_data)
        
        # Should not have logged anything
        assert len(reward_logger.reward_history) == 0
        assert reward_logger.total_rewards_logged == 0
        
    logger.info("✅ Disabled logging test passed")

def run_all_tests():
    """Run all reward logging tests."""
    logger.info("🚀 Starting reward logging tests...")
    
    try:
        test_basic_reward_logging()
        test_performance_bonus_logging()
        test_episode_reward_tracking()
        test_reward_statistics()
        test_save_load_logs()
        test_report_generation()
        test_trend_analysis()
        test_disabled_logging()
        
        logger.info("🎉 All reward logging tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)