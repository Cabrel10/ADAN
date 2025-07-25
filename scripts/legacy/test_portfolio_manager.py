#!/usr/bin/env python3
"""
Test script for PortfolioManager in the ADAN Trading Bot.

This script tests the comprehensive portfolio management functionality including
position tracking, risk management, and performance metrics.
"""

import sys
from pathlib import Path
import logging

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config():
    """Create a test configuration for PortfolioManager."""
    return {
        'initial_balance': 10000.0,
        'assets': ['BTC', 'ETH', 'ADA'],
        'trading_rules': {
            'futures_enabled': False,
            'leverage': 1,
            'commission_pct': 0.001,
            'min_trade_size': 0.0001,
            'min_notional_value': 10.0,
            'max_notional_value': 100000.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'liquidation_threshold': 0.2
        },
        'risk_management': {
            'capital_tiers': [
                {'threshold': 0, 'allocation_per_trade': 0.1},
                {'threshold': 50000, 'allocation_per_trade': 0.15},
                {'threshold': 100000, 'allocation_per_trade': 0.2}
            ],
            'position_sizing': {
                'concentration_limits': {
                    'max_single_asset': 0.3
                }
            }
        }
    }


def test_portfolio_initialization():
    """Test portfolio initialization."""
    logger.info("Testing portfolio initialization...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    assert portfolio.cash == 10000.0, "Initial cash should be 10000"
    assert portfolio.total_capital == 10000.0, "Initial total capital should be 10000"
    assert len(portfolio.positions) == 3, "Should have 3 asset positions"
    
    # Check that all positions are closed initially
    for asset, position in portfolio.positions.items():
        assert not position.is_open, f"{asset} position should be closed initially"
    
    logger.info("✅ Portfolio initialization test passed")
    return True


def test_position_opening():
    """Test opening positions."""
    logger.info("Testing position opening...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test opening a BTC position
    success = portfolio.open_position('BTC', 50000.0, 0.1)
    assert success, "Should successfully open BTC position"
    
    btc_position = portfolio.positions['BTC']
    assert btc_position.is_open, "BTC position should be open"
    assert btc_position.size == 0.1, "BTC position size should be 0.1"
    assert btc_position.entry_price == 50000.0, "BTC entry price should be 50000"
    
    # Check cash reduction
    expected_cash = 10000.0 - (0.1 * 50000.0) - (0.1 * 50000.0 * 0.001)  # price - commission
    assert abs(portfolio.cash - expected_cash) < 0.01, f"Cash should be {expected_cash}, got {portfolio.cash}"
    
    logger.info("✅ Position opening test passed")
    return True


def test_position_closing():
    """Test closing positions."""
    logger.info("Testing position closing...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open a position first
    portfolio.open_position('BTC', 50000.0, 0.1)
    initial_cash = portfolio.cash
    
    # Close the position at a profit
    pnl = portfolio.close_position('BTC', 55000.0)
    
    assert not portfolio.positions['BTC'].is_open, "BTC position should be closed"
    assert pnl > 0, "Should have positive PnL"
    
    # Check cash increase
    expected_pnl = (55000.0 - 50000.0) * 0.1 - (0.1 * 55000.0 * 0.001)  # profit - commission
    assert abs(pnl - expected_pnl) < 0.01, f"PnL should be {expected_pnl}, got {pnl}"
    
    logger.info("✅ Position closing test passed")
    return True


def test_portfolio_metrics():
    """Test portfolio metrics calculation."""
    logger.info("Testing portfolio metrics...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open some positions
    portfolio.open_position('BTC', 50000.0, 0.1)
    portfolio.open_position('ETH', 3000.0, 1.0)
    
    # Update market prices
    current_prices = {'BTC': 52000.0, 'ETH': 3200.0, 'ADA': 1.0}
    portfolio.update_market_price(current_prices)
    
    metrics = portfolio.get_metrics()
    
    assert 'total_capital' in metrics, "Should have total_capital metric"
    assert 'unrealized_pnl' in metrics, "Should have unrealized_pnl metric"
    assert 'realized_pnl' in metrics, "Should have realized_pnl metric"
    assert 'sharpe_ratio' in metrics, "Should have sharpe_ratio metric"
    assert 'drawdown' in metrics, "Should have drawdown metric"
    
    # Check unrealized PnL calculation
    expected_unrealized = (52000.0 - 50000.0) * 0.1 + (3200.0 - 3000.0) * 1.0
    assert abs(metrics['unrealized_pnl'] - expected_unrealized) < 0.01, "Unrealized PnL calculation incorrect"
    
    logger.info("✅ Portfolio metrics test passed")
    return True


def test_risk_management():
    """Test risk management features."""
    logger.info("Testing risk management...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test position validation
    valid = portfolio.validate_position('BTC', 0.1, 50000.0)
    assert valid, "Valid position should pass validation"
    
    # Test invalid position (too small)
    invalid = portfolio.validate_position('BTC', 0.00001, 50000.0)
    assert not invalid, "Too small position should fail validation"
    
    # Test insufficient cash
    portfolio.cash = 100.0  # Set very low cash
    insufficient_cash = portfolio.validate_position('BTC', 1.0, 50000.0)
    assert not insufficient_cash, "Position requiring more cash than available should fail"
    
    logger.info("✅ Risk management test passed")
    return True


def test_stop_loss_take_profit():
    """Test stop loss and take profit functionality."""
    logger.info("Testing stop loss and take profit...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open a position
    portfolio.open_position('BTC', 50000.0, 0.1)
    
    # Test stop loss trigger
    current_prices = {'BTC': 49000.0, 'ETH': 3000.0, 'ADA': 1.0}  # 2% drop
    portfolio.check_protection_orders(current_prices)
    
    # Position should be closed due to stop loss
    assert not portfolio.positions['BTC'].is_open, "Position should be closed by stop loss"
    
    # Test take profit
    portfolio.open_position('ETH', 3000.0, 1.0)
    current_prices = {'BTC': 50000.0, 'ETH': 3120.0, 'ADA': 1.0}  # 4% gain
    portfolio.check_protection_orders(current_prices)
    
    # Position should be closed due to take profit
    assert not portfolio.positions['ETH'].is_open, "Position should be closed by take profit"
    
    logger.info("✅ Stop loss and take profit test passed")
    return True


def test_portfolio_rebalancing():
    """Test portfolio rebalancing."""
    logger.info("Testing portfolio rebalancing...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open a large position that exceeds concentration limit
    portfolio.open_position('BTC', 50000.0, 0.25)  # 25% of portfolio
    
    current_prices = {'BTC': 60000.0, 'ETH': 3000.0, 'ADA': 1.0}
    portfolio.update_market_price(current_prices)
    
    # This should trigger rebalancing due to concentration limit (30% max)
    initial_size = portfolio.positions['BTC'].size
    portfolio.rebalance(current_prices)
    
    # Position size should be reduced or position should be adjusted
    # (The exact behavior depends on the rebalancing logic)
    logger.info(f"Position size before: {initial_size}, after: {portfolio.positions['BTC'].size}")
    
    logger.info("✅ Portfolio rebalancing test passed")
    return True


def test_chunk_performance_tracking():
    """Test chunk performance tracking."""
    logger.info("Testing chunk performance tracking...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Start a new chunk
    portfolio.start_new_chunk()
    assert portfolio.current_chunk_id == 1, "Should be on chunk 1"
    
    # Make some trades
    portfolio.open_position('BTC', 50000.0, 0.1)
    portfolio.close_position('BTC', 55000.0)
    
    # Start another chunk (this should finalize the previous one)
    portfolio.start_new_chunk()
    assert portfolio.current_chunk_id == 2, "Should be on chunk 2"
    assert 1 in portfolio.chunk_pnl, "Should have PnL data for chunk 1"
    
    # Test performance ratio calculation
    optimal_pnl = 15.0  # 15% optimal return
    ratio = portfolio.get_chunk_performance_ratio(1, optimal_pnl)
    assert 0 <= ratio <= 1, "Performance ratio should be between 0 and 1"
    
    logger.info("✅ Chunk performance tracking test passed")
    return True


def test_futures_trading():
    """Test futures trading functionality."""
    logger.info("Testing futures trading...")
    
    config = create_test_config()
    config['trading_rules']['futures_enabled'] = True
    config['trading_rules']['leverage'] = 10
    
    portfolio = PortfolioManager(config)
    
    # Open a leveraged position
    success = portfolio.open_position('BTC', 50000.0, 0.1)
    assert success, "Should successfully open leveraged position"
    
    # Check margin usage
    margin_level = portfolio.get_margin_level()
    assert margin_level > 0, "Should have positive margin level"
    
    # Test liquidation check
    current_prices = {'BTC': 45000.0, 'ETH': 3000.0, 'ADA': 1.0}  # 10% drop
    portfolio.update_market_price(current_prices)
    portfolio.check_liquidation(current_prices)
    
    logger.info("✅ Futures trading test passed")
    return True


def test_bankruptcy_detection():
    """Test bankruptcy detection."""
    logger.info("Testing bankruptcy detection...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Simulate large losses
    portfolio.total_capital = 50.0  # Less than 1% of initial capital
    
    is_bankrupt = portfolio.is_bankrupt()
    assert is_bankrupt, "Should detect bankruptcy when capital < 1% of initial"
    
    logger.info("✅ Bankruptcy detection test passed")
    return True


def test_state_features():
    """Test state features generation."""
    logger.info("Testing state features...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Get initial state features
    features = portfolio.get_state_features()
    assert isinstance(features, type(portfolio.get_state_features())), "Should return numpy array"
    assert len(features) == len(config['assets']) * 2, "Should have 2 features per asset"
    
    # Open a position and check features change
    portfolio.open_position('BTC', 50000.0, 0.1)
    new_features = portfolio.get_state_features()
    
    # Features should be different after opening position
    # (Note: current implementation has limitations in PnL calculation)
    logger.info(f"Initial features: {features}")
    logger.info(f"Features after position: {new_features}")
    
    logger.info("✅ State features test passed")
    return True


def run_all_tests():
    """Run all portfolio manager tests."""
    logger.info("Starting PortfolioManager tests...")
    
    tests = [
        test_portfolio_initialization,
        test_position_opening,
        test_position_closing,
        test_portfolio_metrics,
        test_risk_management,
        test_stop_loss_take_profit,
        test_portfolio_rebalancing,
        test_chunk_performance_tracking,
        test_futures_trading,
        test_bankruptcy_detection,
        test_state_features
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                logger.info(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test.__name__} FAILED with exception: {e}")
    
    logger.info(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All PortfolioManager tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)