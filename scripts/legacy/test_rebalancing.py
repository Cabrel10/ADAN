#!/usr/bin/env python3
"""
Test script for portfolio rebalancing functionality.

This script tests:
1. Basic rebalancing when concentration limits are exceeded
2. Multi-asset rebalancing scenarios
3. Rebalancing with different market conditions
4. Edge cases and error handling
5. Performance impact of rebalancing
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
        'assets': ['BTC', 'ETH', 'ADA'],
        'trading_rules': {
            'futures_enabled': False,
            'leverage': 1,
            'commission_pct': 0.001,
            'min_trade_size': 0.001,
            'min_notional_value': 10.0,
            'max_notional_value': 8000.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'liquidation_threshold': 0.2
        },
        'risk_management': {
            'capital_tiers': [
                {'threshold': 0, 'allocation_per_trade': 0.2}
            ],
            'position_sizing': {
                'concentration_limits': {
                    'max_single_asset': 0.4,  # 40% max per asset
                    'max_total_exposure': 0.8  # 80% max total exposure
                }
            }
        }
    }

def test_basic_rebalancing():
    """Test basic rebalancing when a single asset exceeds concentration limits."""
    logger.info("Testing basic rebalancing...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open a large BTC position that exceeds the 40% limit
    btc_price = 50000.0
    btc_size = 0.12  # This will be ~60% of portfolio
    portfolio.open_position('BTC', btc_price, btc_size)
    
    current_prices = {'BTC': btc_price, 'ETH': 3000.0, 'ADA': 1.0}
    portfolio.update_market_price(current_prices)
    
    # Check initial allocation
    btc_value = portfolio.positions['BTC'].size * current_prices['BTC']
    initial_allocation = btc_value / portfolio.get_portfolio_value()
    
    logger.info(f"Initial BTC allocation: {initial_allocation:.2%}")
    assert initial_allocation > 0.4, f"Initial allocation should exceed 40%, got {initial_allocation:.2%}"
    
    # Perform rebalancing
    portfolio.rebalance(current_prices)
    
    # Check allocation after rebalancing
    if portfolio.positions['BTC'].is_open:
        btc_value_after = portfolio.positions['BTC'].size * current_prices['BTC']
        final_allocation = btc_value_after / portfolio.get_portfolio_value()
        logger.info(f"Final BTC allocation: {final_allocation:.2%}")
        assert final_allocation <= 0.41, f"Final allocation should be <= 40%, got {final_allocation:.2%}"
    else:
        logger.info("BTC position was fully closed during rebalancing")
    
    logger.info("✅ Basic rebalancing test passed")

def test_multi_asset_rebalancing():
    """Test rebalancing with multiple assets exceeding limits."""
    logger.info("Testing multi-asset rebalancing...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open positions in multiple assets that exceed limits
    portfolio.open_position('BTC', 50000.0, 0.1)   # ~50% of portfolio
    portfolio.open_position('ETH', 3000.0, 1.5)    # ~45% of portfolio
    
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0, 'ADA': 1.0}
    portfolio.update_market_price(current_prices)
    
    # Check initial allocations
    btc_value = portfolio.positions['BTC'].size * current_prices['BTC']
    eth_value = portfolio.positions['ETH'].size * current_prices['ETH']
    total_value = portfolio.get_portfolio_value()
    
    btc_allocation = btc_value / total_value
    eth_allocation = eth_value / total_value
    
    logger.info(f"Initial allocations - BTC: {btc_allocation:.2%}, ETH: {eth_allocation:.2%}")
    
    # Both should exceed the 40% limit
    assert btc_allocation > 0.4, f"BTC allocation should exceed 40%, got {btc_allocation:.2%}"
    assert eth_allocation > 0.4, f"ETH allocation should exceed 40%, got {eth_allocation:.2%}"
    
    # Perform rebalancing
    portfolio.rebalance(current_prices)
    
    # Check allocations after rebalancing
    total_value_after = portfolio.get_portfolio_value()
    
    if portfolio.positions['BTC'].is_open:
        btc_value_after = portfolio.positions['BTC'].size * current_prices['BTC']
        btc_allocation_after = btc_value_after / total_value_after
        logger.info(f"Final BTC allocation: {btc_allocation_after:.2%}")
        assert btc_allocation_after <= 0.41, f"BTC allocation should be <= 40%, got {btc_allocation_after:.2%}"
    
    if portfolio.positions['ETH'].is_open:
        eth_value_after = portfolio.positions['ETH'].size * current_prices['ETH']
        eth_allocation_after = eth_value_after / total_value_after
        logger.info(f"Final ETH allocation: {eth_allocation_after:.2%}")
        assert eth_allocation_after <= 0.41, f"ETH allocation should be <= 40%, got {eth_allocation_after:.2%}"
    
    logger.info("✅ Multi-asset rebalancing test passed")

def test_rebalancing_with_price_changes():
    """Test rebalancing behavior when prices change significantly."""
    logger.info("Testing rebalancing with price changes...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open positions at initial prices
    initial_prices = {'BTC': 50000.0, 'ETH': 3000.0, 'ADA': 1.0}
    portfolio.open_position('BTC', initial_prices['BTC'], 0.08)
    portfolio.open_position('ETH', initial_prices['ETH'], 1.0)
    
    # Simulate significant price increase for BTC
    new_prices = {'BTC': 75000.0, 'ETH': 3000.0, 'ADA': 1.0}  # BTC up 50%
    portfolio.update_market_price(new_prices)
    
    # Check if BTC allocation now exceeds limits due to price increase
    btc_value = portfolio.positions['BTC'].size * new_prices['BTC']
    total_value = portfolio.get_portfolio_value()
    btc_allocation = btc_value / total_value
    
    logger.info(f"BTC allocation after price increase: {btc_allocation:.2%}")
    
    if btc_allocation > 0.4:
        # Perform rebalancing
        portfolio.rebalance(new_prices)
        
        # Check allocation after rebalancing
        if portfolio.positions['BTC'].is_open:
            btc_value_after = portfolio.positions['BTC'].size * new_prices['BTC']
            btc_allocation_after = btc_value_after / portfolio.get_portfolio_value()
            logger.info(f"BTC allocation after rebalancing: {btc_allocation_after:.2%}")
            assert btc_allocation_after <= 0.41, f"BTC allocation should be <= 40%, got {btc_allocation_after:.2%}"
    
    logger.info("✅ Rebalancing with price changes test passed")

def test_rebalancing_edge_cases():
    """Test rebalancing edge cases and error handling."""
    logger.info("Testing rebalancing edge cases...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test 1: Rebalancing with no positions
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0, 'ADA': 1.0}
    portfolio.rebalance(current_prices)  # Should not crash
    
    # Test 2: Rebalancing with empty price data
    portfolio.open_position('BTC', 50000.0, 0.1)
    portfolio.rebalance({})  # Should handle gracefully
    
    # Test 3: Rebalancing with missing price for an asset
    incomplete_prices = {'BTC': 50000.0, 'ADA': 1.0}  # Missing ETH price
    portfolio.rebalance(incomplete_prices)  # Should handle gracefully
    
    # Test 4: Rebalancing with zero portfolio value
    portfolio.cash = 0.0
    portfolio.positions['BTC'].close()
    portfolio.rebalance(current_prices)  # Should handle gracefully
    
    logger.info("✅ Rebalancing edge cases test passed")

def test_rebalancing_performance_impact():
    """Test the performance impact of rebalancing operations."""
    logger.info("Testing rebalancing performance impact...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open a position that will trigger rebalancing
    portfolio.open_position('BTC', 50000.0, 0.12)  # ~60% allocation
    
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0, 'ADA': 1.0}
    portfolio.update_market_price(current_prices)
    
    # Record portfolio value before rebalancing
    value_before = portfolio.get_portfolio_value()
    realized_pnl_before = portfolio.realized_pnl
    
    # Perform rebalancing
    portfolio.rebalance(current_prices)
    
    # Record portfolio value after rebalancing
    value_after = portfolio.get_portfolio_value()
    realized_pnl_after = portfolio.realized_pnl
    
    # Calculate the cost of rebalancing (should include commissions)
    rebalancing_cost = value_before - value_after
    pnl_change = realized_pnl_after - realized_pnl_before
    
    logger.info(f"Portfolio value before rebalancing: ${value_before:.2f}")
    logger.info(f"Portfolio value after rebalancing: ${value_after:.2f}")
    logger.info(f"Rebalancing cost: ${rebalancing_cost:.2f}")
    logger.info(f"PnL change from rebalancing: ${pnl_change:.2f}")
    
    # Rebalancing should have some cost due to commissions
    assert rebalancing_cost >= 0, "Rebalancing cost should be non-negative"
    
    # The cost should be reasonable (less than 1% of portfolio value)
    cost_percentage = rebalancing_cost / value_before
    assert cost_percentage < 0.01, f"Rebalancing cost should be < 1%, got {cost_percentage:.2%}"
    
    logger.info("✅ Rebalancing performance impact test passed")

def test_rebalancing_with_futures():
    """Test rebalancing behavior with futures trading enabled."""
    logger.info("Testing rebalancing with futures...")
    
    config = create_test_config()
    config['trading_rules']['futures_enabled'] = True
    config['trading_rules']['leverage'] = 3
    config['trading_rules']['futures_commission_pct'] = 0.0005
    
    portfolio = PortfolioManager(config)
    
    # Open leveraged positions
    portfolio.open_position('BTC', 50000.0, 0.3)  # With 3x leverage, this uses less margin
    
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0, 'ADA': 1.0}
    portfolio.update_market_price(current_prices)
    
    # Check margin level
    margin_level = portfolio.get_margin_level()
    logger.info(f"Margin level before rebalancing: {margin_level:.2%}")
    
    # Perform rebalancing
    portfolio.rebalance(current_prices)
    
    # Check margin level after rebalancing
    margin_level_after = portfolio.get_margin_level()
    logger.info(f"Margin level after rebalancing: {margin_level_after:.2%}")
    
    # Margin level should be reasonable
    assert margin_level_after < 0.8, f"Margin level should be < 80%, got {margin_level_after:.2%}"
    
    logger.info("✅ Rebalancing with futures test passed")

def test_concentration_limits_enforcement():
    """Test that concentration limits are properly enforced during rebalancing."""
    logger.info("Testing concentration limits enforcement...")
    
    config = create_test_config()
    # Set stricter concentration limits
    config['risk_management']['position_sizing']['concentration_limits']['max_single_asset'] = 0.3  # 30%
    
    portfolio = PortfolioManager(config)
    
    # Open a position that significantly exceeds the limit
    portfolio.open_position('BTC', 50000.0, 0.15)  # ~75% of portfolio
    
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0, 'ADA': 1.0}
    portfolio.update_market_price(current_prices)
    
    # Check initial allocation
    btc_value = portfolio.positions['BTC'].size * current_prices['BTC']
    initial_allocation = btc_value / portfolio.get_portfolio_value()
    logger.info(f"Initial BTC allocation: {initial_allocation:.2%}")
    
    # Perform rebalancing
    portfolio.rebalance(current_prices)
    
    # Check that the limit is enforced
    if portfolio.positions['BTC'].is_open:
        btc_value_after = portfolio.positions['BTC'].size * current_prices['BTC']
        final_allocation = btc_value_after / portfolio.get_portfolio_value()
        logger.info(f"Final BTC allocation: {final_allocation:.2%}")
        assert final_allocation <= 0.31, f"Final allocation should be <= 30%, got {final_allocation:.2%}"
    
    logger.info("✅ Concentration limits enforcement test passed")

def run_all_tests():
    """Run all rebalancing tests."""
    logger.info("🚀 Starting rebalancing tests...")
    
    try:
        test_basic_rebalancing()
        test_multi_asset_rebalancing()
        test_rebalancing_with_price_changes()
        test_rebalancing_edge_cases()
        test_rebalancing_performance_impact()
        test_rebalancing_with_futures()
        test_concentration_limits_enforcement()
        
        logger.info("🎉 All rebalancing tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)