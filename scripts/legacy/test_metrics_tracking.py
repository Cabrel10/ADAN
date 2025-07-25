#!/usr/bin/env python3
"""
Test script for the MetricsTracker component.

This script tests:
1. Basic metrics tracking functionality
2. Different types of metrics (trading, learning, execution, market)
3. Trade logging and episode tracking
4. Metrics aggregation and analysis
5. Saving and loading metrics
6. Performance and memory management
"""

import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.common.metrics_tracker import MetricsTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config(temp_dir: str) -> Dict[str, Any]:
    """Create a test configuration for the MetricsTracker."""
    return {
        'metrics_tracking': {
            'enabled': True,
            'save_interval': 10,  # Save every 10 updates for testing
            'history_length': 100,  # Keep last 100 records
            'base_path': temp_dir
        }
    }

def test_basic_metrics_tracking():
    """Test basic metrics tracking functionality."""
    logger.info("Testing basic metrics tracking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        tracker = MetricsTracker(config)
        
        # Test trading metrics
        trading_metrics = {
            'pnl': 100.0,
            'realized_pnl': 50.0,
            'unrealized_pnl': 50.0,
            'total_capital': 10100.0,
            'drawdown': -0.02,
            'sharpe_ratio': 1.5,
            'win_rate': 0.6,
            'trade_count': 10
        }
        
        tracker.update_trading_metrics(trading_metrics)
        
        # Test learning metrics
        learning_metrics = {
            'reward': 0.5,
            'loss': 0.1,
            'epsilon': 0.1,
            'learning_rate': 0.001
        }
        
        tracker.update_learning_metrics(learning_metrics)
        
        # Test execution metrics
        execution_metrics = {
            'order_latency': 0.05,
            'slippage': 0.001,
            'commission': 5.0
        }
        
        tracker.update_execution_metrics(execution_metrics)
        
        # Test market metrics
        market_metrics = {
            'volatility': 0.02,
            'BTC_price': 50000.0,
            'ETH_price': 3000.0
        }
        
        tracker.update_market_metrics(market_metrics)
        
        # Call step to increment update counter
        tracker.step()
        
        # Verify metrics were stored
        current_metrics = tracker.get_current_metrics()
        assert 'total_updates' in current_metrics
        assert current_metrics['total_updates'] > 0
        
        # Check specific metric history
        pnl_history = tracker.get_metric_history('trading.pnl')
        assert len(pnl_history) == 1
        assert pnl_history[0]['value'] == 100.0
        
    logger.info("✅ Basic metrics tracking test passed")

def test_trade_logging():
    """Test trade logging functionality."""
    logger.info("Testing trade logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        tracker = MetricsTracker(config)
        
        # Log some trades
        trades = [
            {'asset': 'BTC', 'action': 'buy', 'size': 0.1, 'price': 50000.0, 'pnl': 100.0, 'commission': 5.0},
            {'asset': 'ETH', 'action': 'sell', 'size': 1.0, 'price': 3000.0, 'pnl': -50.0, 'commission': 3.0},
            {'asset': 'BTC', 'action': 'sell', 'size': 0.05, 'price': 51000.0, 'pnl': 200.0, 'commission': 2.5}
        ]
        
        for trade in trades:
            tracker.log_trade(trade)
        
        # Verify trades were logged
        assert len(tracker.trade_history) == 3
        
        # Check trade details
        logged_trade = tracker.trade_history[0]
        assert logged_trade['asset'] == 'BTC'
        assert logged_trade['pnl'] == 100.0
        
        # Check current metrics include trade info
        current_metrics = tracker.get_current_metrics()
        assert 'recent_win_rate' in current_metrics
        assert 'recent_avg_pnl' in current_metrics
        
        # Win rate should be 2/3 (2 winning trades out of 3)
        expected_win_rate = 2.0 / 3.0
        assert abs(current_metrics['recent_win_rate'] - expected_win_rate) < 0.01
        
    logger.info("✅ Trade logging test passed")

def test_episode_tracking():
    """Test episode tracking functionality."""
    logger.info("Testing episode tracking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        tracker = MetricsTracker(config)
        
        # Start episode
        tracker.start_episode(1)
        assert tracker.current_episode == 1
        assert tracker.current_step == 0
        
        # Simulate some steps
        for i in range(10):
            tracker.step()
        
        assert tracker.current_step == 10
        
        # End episode
        episode_metrics = {
            'total_reward': 5.0,
            'total_pnl': 100.0,
            'episode_length': 10
        }
        
        tracker.end_episode(episode_metrics)
        
        # Check episode metrics were recorded
        episode_reward_history = tracker.get_metric_history('episodes.total_reward')
        assert len(episode_reward_history) == 1
        assert episode_reward_history[0]['value'] == 5.0
        
    logger.info("✅ Episode tracking test passed")

def test_metrics_aggregation():
    """Test metrics aggregation and analysis."""
    logger.info("Testing metrics aggregation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        tracker = MetricsTracker(config)
        
        # Add multiple reward values to test trend calculation
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for reward in rewards:
            tracker.update_learning_metrics({'reward': reward})
        
        # Get current metrics
        current_metrics = tracker.get_current_metrics()
        
        # Check average reward
        assert 'avg_reward_recent' in current_metrics
        expected_avg = np.mean(rewards)
        assert abs(current_metrics['avg_reward_recent'] - expected_avg) < 0.01
        
        # Check trend (should be 'up' for increasing values)
        assert current_metrics['reward_trend'] == 'up'
        
        # Test with decreasing values - create a new tracker for cleaner test
        tracker2 = MetricsTracker(config)
        decreasing_rewards = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for reward in decreasing_rewards:
            tracker2.update_learning_metrics({'reward': reward})
        
        current_metrics2 = tracker2.get_current_metrics()
        # Trend should now be 'down'
        assert current_metrics2['reward_trend'] == 'down'
        
    logger.info("✅ Metrics aggregation test passed")

def test_save_load_metrics():
    """Test saving and loading metrics."""
    logger.info("Testing save/load metrics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        tracker = MetricsTracker(config)
        
        # Add some data
        tracker.update_trading_metrics({'pnl': 100.0, 'trade_count': 5})
        tracker.log_trade({'asset': 'BTC', 'pnl': 50.0, 'commission': 2.0})
        tracker.start_episode(1)
        
        # Save metrics
        filename = 'test_metrics.json'
        tracker.save_metrics(filename)
        
        # Verify file was created
        filepath = Path(temp_dir) / filename
        assert filepath.exists()
        
        # Create new tracker and load metrics
        tracker2 = MetricsTracker(config)
        success = tracker2.load_metrics(filename)
        
        assert success
        assert len(tracker2.trade_history) > 0
        assert tracker2.current_episode == 1
        
    logger.info("✅ Save/load metrics test passed")

def test_performance_and_memory():
    """Test performance and memory management."""
    logger.info("Testing performance and memory management...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        # Set smaller history length for testing
        config['metrics_tracking']['history_length'] = 50
        tracker = MetricsTracker(config)
        
        # Add many metrics to test memory management
        start_time = time.time()
        
        for i in range(100):
            tracker.update_trading_metrics({
                'pnl': float(i),
                'trade_count': i,
                'total_capital': 10000.0 + i
            })
            tracker.step()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 1.0, f"Performance test took too long: {duration:.2f}s"
        
        # Check that history length is respected
        pnl_history = tracker.get_metric_history('trading.pnl')
        assert len(pnl_history) <= 50, f"History length exceeded: {len(pnl_history)}"
        
        # Check that latest values are preserved
        latest_pnl = pnl_history[-1]['value']
        assert latest_pnl == 99.0, f"Latest value not preserved: {latest_pnl}"
        
    logger.info("✅ Performance and memory test passed")

def test_disabled_tracking():
    """Test behavior when tracking is disabled."""
    logger.info("Testing disabled tracking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        config['metrics_tracking']['enabled'] = False
        tracker = MetricsTracker(config)
        
        # Try to add metrics
        tracker.update_trading_metrics({'pnl': 100.0})
        tracker.log_trade({'asset': 'BTC', 'pnl': 50.0})
        
        # Should not have stored anything
        assert len(tracker.trade_history) == 0
        assert len(tracker.metrics) == 0
        
        # Save should not create files
        tracker.save_metrics('test.json')
        filepath = Path(temp_dir) / 'test.json'
        assert not filepath.exists()
        
    logger.info("✅ Disabled tracking test passed")

def test_metric_summaries():
    """Test metric summary generation."""
    logger.info("Testing metric summaries...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config(temp_dir)
        tracker = MetricsTracker(config)
        
        # Add some data with known statistics
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            tracker.update_trading_metrics({'pnl': value})
        
        # Generate summaries
        summaries = tracker._generate_metric_summaries()
        
        # Check PnL summary
        pnl_summary = summaries['trading.pnl']
        assert pnl_summary['count'] == 5
        assert pnl_summary['mean'] == 3.0
        assert pnl_summary['min'] == 1.0
        assert pnl_summary['max'] == 5.0
        assert pnl_summary['latest'] == 5.0
        
    logger.info("✅ Metric summaries test passed")

def run_all_tests():
    """Run all metrics tracking tests."""
    logger.info("🚀 Starting metrics tracking tests...")
    
    try:
        test_basic_metrics_tracking()
        test_trade_logging()
        test_episode_tracking()
        test_metrics_aggregation()
        test_save_load_metrics()
        test_performance_and_memory()
        test_disabled_tracking()
        test_metric_summaries()
        
        logger.info("🎉 All metrics tracking tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)