#!/usr/bin/env python3
"""
Test script for position tracking functionality in PortfolioManager.

This script tests:
1. Position opening and closing
2. Position state tracking
3. Portfolio value updates
4. Risk metrics calculation
5. Rebalancing functionality
6. Position validation
"""

import sys
import os
import logging
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config() -> Dict[str, Any]:
    """Create a test configuration for the PortfolioManager."""
    return {
        'initial_balance': 10000.0,
        'assets': ['BTC', 'ETH'],
        'trading_rules': {
            'futures_enabled': False,
            'leverage': 1,
            'commission_pct': 0.001,
            'min_trade_size': 0.001,
            'min_notional_value': 10.0,
            'max_notional_value': 5000.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'liquidation_threshold': 0.2
        },
        'risk_management': {
            'capital_tiers': [
                {'threshold': 0, 'allocation_per_trade': 0.1},
                {'threshold': 20000, 'allocation_per_trade': 0.15},
                {'threshold': 50000, 'allocation_per_trade': 0.2}
            ],
            'position_sizing': {
                'concentration_limits': {
                    'max_single_asset': 0.5
                }
            }
        }
    }

def test_position_opening_closing():
    """Test basic position opening and closing functionality."""
    logger.info("Testing position opening and closing...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test initial state
    assert portfolio.get_portfolio_value() == 10000.0
    assert not portfolio.positions['BTC'].is_open
    assert not portfolio.positions['ETH'].is_open
    
    # Test opening a position
    btc_price = 50000.0
    btc_size = 0.1
    success = portfolio.open_position('BTC', btc_price, btc_size)
    
    assert success, "Position should open successfully"
    assert portfolio.positions['BTC'].is_open, "BTC position should be open"
    assert portfolio.positions['BTC'].size == btc_size
    assert portfolio.positions['BTC'].entry_price == btc_price
    
    # Check cash reduction
    expected_cash = 10000.0 - (btc_size * btc_price) - (btc_size * btc_price * 0.001)
    assert abs(portfolio.cash - expected_cash) < 0.01, f"Cash should be {expected_cash}, got {portfolio.cash}"
    
    # Test closing the position
    close_price = 52000.0
    pnl = portfolio.close_position('BTC', close_price)
    
    assert not portfolio.positions['BTC'].is_open, "BTC position should be closed"
    expected_pnl = (close_price - btc_price) * btc_size - (btc_size * close_price * 0.001)
    assert abs(pnl - expected_pnl) < 0.01, f"PnL should be {expected_pnl}, got {pnl}"
    
    logger.info("✅ Position opening and closing test passed")

def test_portfolio_value_tracking():
    """Test portfolio value tracking with market price updates."""
    logger.info("Testing portfolio value tracking...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open positions
    portfolio.open_position('BTC', 50000.0, 0.1)
    portfolio.open_position('ETH', 3000.0, 1.0)
    
    # Update market prices
    current_prices = {'BTC': 55000.0, 'ETH': 3200.0}
    portfolio.update_market_price(current_prices)
    
    # Calculate expected portfolio value
    btc_value = 0.1 * 55000.0  # 5500
    eth_value = 1.0 * 3200.0   # 3200
    total_market_value = btc_value + eth_value  # 8700
    
    # Account for commissions paid when opening positions
    btc_commission = 0.1 * 50000.0 * 0.001  # 5.0
    eth_commission = 1.0 * 3000.0 * 0.001   # 3.0
    total_commissions = btc_commission + eth_commission  # 8.0
    
    # Cash remaining after opening positions
    remaining_cash = 10000.0 - (0.1 * 50000.0) - (1.0 * 3000.0) - total_commissions  # 1992.0
    
    expected_portfolio_value = remaining_cash + total_market_value
    
    assert abs(portfolio.get_portfolio_value() - expected_portfolio_value) < 1.0, \
        f"Portfolio value should be around {expected_portfolio_value}, got {portfolio.get_portfolio_value()}"
    
    logger.info("✅ Portfolio value tracking test passed")

def test_risk_metrics_calculation():
    """Test risk metrics calculation."""
    logger.info("Testing risk metrics calculation...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Simulate some trading history
    prices = [50000.0, 51000.0, 49000.0, 52000.0, 48000.0]
    
    for i, price in enumerate(prices):
        if i > 0:
            portfolio.close_position('BTC', price)
        if i < len(prices) - 1:
            portfolio.open_position('BTC', price, 0.1)
        
        current_prices = {'BTC': price, 'ETH': 3000.0}
        portfolio.update_market_price(current_prices)
    
    metrics = portfolio.get_metrics()
    
    # Check that metrics are calculated
    assert 'drawdown' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'var' in metrics
    assert 'cvar' in metrics
    
    # Drawdown should be negative or zero
    assert metrics['drawdown'] <= 0, f"Drawdown should be <= 0, got {metrics['drawdown']}"
    
    logger.info("✅ Risk metrics calculation test passed")

def test_position_validation():
    """Test position validation functionality."""
    logger.info("Testing position validation...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test valid position
    assert portfolio.validate_position('BTC', 0.1, 50000.0), "Valid position should pass validation"
    
    # Test invalid size
    assert not portfolio.validate_position('BTC', 0.0, 50000.0), "Zero size should fail validation"
    assert not portfolio.validate_position('BTC', -0.1, 50000.0), "Negative size should fail validation"
    
    # Test invalid price
    assert not portfolio.validate_position('BTC', 0.1, 0.0), "Zero price should fail validation"
    assert not portfolio.validate_position('BTC', 0.1, -1000.0), "Negative price should fail validation"
    
    # Test minimum trade size
    assert not portfolio.validate_position('BTC', 0.0001, 50000.0), "Below minimum trade size should fail"
    
    # Test minimum notional value
    assert not portfolio.validate_position('BTC', 0.0001, 10.0), "Below minimum notional value should fail"
    
    # Test maximum notional value
    assert not portfolio.validate_position('BTC', 1.0, 10000.0), "Above maximum notional value should fail"
    
    # Test insufficient cash
    portfolio.cash = 100.0  # Set very low cash
    assert not portfolio.validate_position('BTC', 0.1, 50000.0), "Insufficient cash should fail validation"
    
    logger.info("✅ Position validation test passed")

def test_rebalancing():
    """Test portfolio rebalancing functionality."""
    logger.info("Testing portfolio rebalancing...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open a large position that exceeds concentration limits
    portfolio.open_position('BTC', 50000.0, 0.15)  # 75% of portfolio
    
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0}
    portfolio.update_market_price(current_prices)
    
    # Check initial allocation
    btc_value = portfolio.positions['BTC'].size * current_prices['BTC']
    initial_allocation = btc_value / portfolio.get_portfolio_value()
    
    assert initial_allocation > 0.5, "Initial allocation should exceed concentration limit"
    
    # Perform rebalancing
    portfolio.rebalance(current_prices)
    
    # Check allocation after rebalancing
    if portfolio.positions['BTC'].is_open:
        btc_value_after = portfolio.positions['BTC'].size * current_prices['BTC']
        final_allocation = btc_value_after / portfolio.get_portfolio_value()
        assert final_allocation <= 0.51, f"Final allocation should be <= 0.5, got {final_allocation}"
    
    logger.info("✅ Rebalancing test passed")

def test_protection_orders():
    """Test stop-loss and take-profit functionality."""
    logger.info("Testing protection orders...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open position
    entry_price = 50000.0
    portfolio.open_position('BTC', entry_price, 0.1)
    
    # Test stop-loss trigger
    stop_loss_price = entry_price * (1 - 0.02)  # 2% stop-loss
    current_prices = {'BTC': stop_loss_price - 100, 'ETH': 3000.0}
    
    portfolio.update_market_price(current_prices)
    
    # Position should be closed due to stop-loss
    assert not portfolio.positions['BTC'].is_open, "Position should be closed due to stop-loss"
    
    # Test take-profit trigger
    portfolio.open_position('BTC', entry_price, 0.1)
    take_profit_price = entry_price * (1 + 0.04)  # 4% take-profit
    current_prices = {'BTC': take_profit_price + 100, 'ETH': 3000.0}
    
    portfolio.update_market_price(current_prices)
    
    # Position should be closed due to take-profit
    assert not portfolio.positions['BTC'].is_open, "Position should be closed due to take-profit"
    
    logger.info("✅ Protection orders test passed")

def test_chunk_tracking():
    """Test chunk-based PnL tracking."""
    logger.info("Testing chunk tracking...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Start first chunk
    portfolio.start_new_chunk()
    assert portfolio.current_chunk_id == 1
    
    # Simulate some trading
    portfolio.open_position('BTC', 50000.0, 0.1)
    current_prices = {'BTC': 52000.0, 'ETH': 3000.0}
    portfolio.update_market_price(current_prices)
    portfolio.close_position('BTC', 52000.0)
    
    # Start second chunk
    portfolio.start_new_chunk()
    assert portfolio.current_chunk_id == 2
    
    # Check that first chunk PnL was recorded
    assert 1 in portfolio.chunk_pnl, "First chunk PnL should be recorded"
    
    chunk_1_data = portfolio.chunk_pnl[1]
    assert 'pnl_pct' in chunk_1_data
    assert 'n_trades' in chunk_1_data
    assert chunk_1_data['n_trades'] > 0, "Should have recorded trades in chunk 1"
    
    logger.info("✅ Chunk tracking test passed")

def run_all_tests():
    """Run all position tracking tests."""
    logger.info("🚀 Starting position tracking tests...")
    
    try:
        test_position_opening_closing()
        test_portfolio_value_tracking()
        test_risk_metrics_calculation()
        test_position_validation()
        test_rebalancing()
        test_protection_orders()
        test_chunk_tracking()
        
        logger.info("🎉 All position tracking tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)