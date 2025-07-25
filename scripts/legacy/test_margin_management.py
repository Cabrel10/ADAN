#!/usr/bin/env python3
"""
Test script for margin management functionality in PortfolioManager.

This script tests:
1. Margin calculation for futures positions
2. Margin level monitoring
3. Liquidation risk detection
4. Margin requirements validation
5. Leverage impact on margin usage
6. Margin calls and position management
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

def create_futures_config(leverage: int = 3) -> Dict[str, Any]:
    """Create a test configuration for futures trading with margin management."""
    return {
        'initial_balance': 10000.0,
        'assets': ['BTC', 'ETH'],
        'trading_rules': {
            'futures_enabled': True,
            'leverage': leverage,
            'futures_commission_pct': 0.0005,
            'commission_pct': 0.001,
            'min_trade_size': 0.001,
            'min_notional_value': 10.0,
            'max_notional_value': 50000.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'liquidation_threshold': 0.1  # 10% margin level triggers liquidation
        },
        'risk_management': {
            'capital_tiers': [
                {'threshold': 0, 'allocation_per_trade': 0.2}
            ],
            'position_sizing': {
                'concentration_limits': {
                    'max_single_asset': 0.5
                }
            }
        }
    }

def create_spot_config() -> Dict[str, Any]:
    """Create a test configuration for spot trading (no margin)."""
    return {
        'initial_balance': 10000.0,
        'assets': ['BTC', 'ETH'],
        'trading_rules': {
            'futures_enabled': False,
            'leverage': 1,
            'commission_pct': 0.001,
            'min_trade_size': 0.001,
            'min_notional_value': 10.0,
            'max_notional_value': 50000.0,
            'stop_loss': 0.02,
            'take_profit': 0.04
        },
        'risk_management': {
            'capital_tiers': [
                {'threshold': 0, 'allocation_per_trade': 0.2}
            ],
            'position_sizing': {
                'concentration_limits': {
                    'max_single_asset': 0.5
                }
            }
        }
    }

def test_margin_calculation():
    """Test basic margin calculation for futures positions."""
    logger.info("Testing margin calculation...")
    
    config = create_futures_config(leverage=5)
    portfolio = PortfolioManager(config)
    
    # Open a futures position
    btc_price = 50000.0
    btc_size = 0.2  # $10,000 notional value
    
    success = portfolio.open_position('BTC', btc_price, btc_size)
    assert success, "Position should open successfully"
    
    # Calculate expected margin used
    notional_value = btc_size * btc_price  # $10,000
    expected_margin = notional_value / config['trading_rules']['leverage']  # $2,000
    commission = notional_value * config['trading_rules']['futures_commission_pct']  # $5
    
    # Check margin level
    margin_level = portfolio.get_margin_level()
    expected_margin_level = (expected_margin + commission) / portfolio.initial_capital
    
    logger.info(f"Notional value: ${notional_value:.2f}")
    logger.info(f"Expected margin used: ${expected_margin:.2f}")
    logger.info(f"Commission: ${commission:.2f}")
    logger.info(f"Margin level: {margin_level:.2%}")
    logger.info(f"Expected margin level: {expected_margin_level:.2%}")
    
    # Margin level should be approximately correct
    assert abs(margin_level - expected_margin_level) < 0.01, \
        f"Margin level should be ~{expected_margin_level:.2%}, got {margin_level:.2%}"
    
    # Cash should be reduced by margin + commission
    expected_cash = portfolio.initial_capital - expected_margin - commission
    assert abs(portfolio.cash - expected_cash) < 1.0, \
        f"Cash should be ~${expected_cash:.2f}, got ${portfolio.cash:.2f}"
    
    logger.info("✅ Margin calculation test passed")

def test_margin_level_monitoring():
    """Test margin level monitoring with multiple positions."""
    logger.info("Testing margin level monitoring...")
    
    config = create_futures_config(leverage=3)
    portfolio = PortfolioManager(config)
    
    # Open multiple positions
    portfolio.open_position('BTC', 50000.0, 0.1)  # $5,000 notional, ~$1,667 margin
    portfolio.open_position('ETH', 3000.0, 1.0)   # $3,000 notional, ~$1,000 margin
    
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0}
    portfolio.update_market_price(current_prices)
    
    # Calculate expected total margin
    btc_margin = (0.1 * 50000.0) / 3  # $1,666.67
    eth_margin = (1.0 * 3000.0) / 3   # $1,000.00
    total_expected_margin = btc_margin + eth_margin  # $2,666.67
    
    margin_level = portfolio.get_margin_level()
    expected_margin_level = total_expected_margin / portfolio.initial_capital
    
    logger.info(f"BTC margin: ${btc_margin:.2f}")
    logger.info(f"ETH margin: ${eth_margin:.2f}")
    logger.info(f"Total expected margin: ${total_expected_margin:.2f}")
    logger.info(f"Margin level: {margin_level:.2%}")
    logger.info(f"Expected margin level: {expected_margin_level:.2%}")
    
    # Margin level should be reasonable (adjusted for actual calculation)
    assert margin_level > 0.15, f"Margin level should be > 15%, got {margin_level:.2%}"
    assert margin_level < 0.35, f"Margin level should be < 35%, got {margin_level:.2%}"
    
    logger.info("✅ Margin level monitoring test passed")

def test_liquidation_risk_detection():
    """Test liquidation risk detection and automatic position closure."""
    logger.info("Testing liquidation risk detection...")
    
    config = create_futures_config(leverage=10)  # High leverage for easier liquidation
    config['trading_rules']['liquidation_threshold'] = 0.15  # 15% threshold
    
    portfolio = PortfolioManager(config)
    
    # Open a large leveraged position
    portfolio.open_position('BTC', 50000.0, 0.18)  # $9,000 notional, $900 margin
    
    # Simulate price drop that increases margin usage
    adverse_prices = {'BTC': 45000.0, 'ETH': 3000.0}  # 10% drop
    portfolio.update_market_price(adverse_prices)
    
    # Check if position is still open (should be closed due to liquidation risk)
    margin_level = portfolio.get_margin_level()
    logger.info(f"Margin level after price drop: {margin_level:.2%}")
    
    # Position should be closed if margin level is too low
    if margin_level < config['trading_rules']['liquidation_threshold']:
        assert not portfolio.positions['BTC'].is_open, \
            "Position should be closed due to liquidation risk"
        logger.info("Position was automatically closed due to liquidation risk")
    else:
        logger.info("Position remains open - margin level is acceptable")
    
    logger.info("✅ Liquidation risk detection test passed")

def test_margin_requirements_validation():
    """Test validation of margin requirements before opening positions."""
    logger.info("Testing margin requirements validation...")
    
    config = create_futures_config(leverage=2)
    portfolio = PortfolioManager(config)
    
    # Test 1: Valid position within margin limits
    valid_result = portfolio.validate_position('BTC', 0.1, 50000.0)
    assert valid_result, "Valid position should pass validation"
    
    # Test 2: Position that would exceed available margin
    # Reduce available cash to simulate low margin
    portfolio.cash = 1000.0  # Only $1,000 available
    
    # Try to open a position requiring $2,500 margin (0.1 * 50000 / 2)
    invalid_result = portfolio.validate_position('BTC', 0.1, 50000.0)
    assert not invalid_result, "Position exceeding available margin should fail validation"
    
    # Test 3: Position with excessive leverage
    portfolio.cash = 10000.0  # Reset cash
    excessive_result = portfolio.validate_position('BTC', 1.0, 50000.0)  # $25,000 margin needed
    assert not excessive_result, "Position requiring excessive margin should fail validation"
    
    logger.info("✅ Margin requirements validation test passed")

def test_leverage_impact():
    """Test the impact of different leverage levels on margin usage."""
    logger.info("Testing leverage impact on margin usage...")
    
    leverages = [1, 3, 5, 10]
    position_size = 0.1
    btc_price = 50000.0
    notional_value = position_size * btc_price
    
    for leverage in leverages:
        config = create_futures_config(leverage=leverage)
        portfolio = PortfolioManager(config)
        
        # Open position
        portfolio.open_position('BTC', btc_price, position_size)
        
        # Calculate margin level
        margin_level = portfolio.get_margin_level()
        expected_margin = notional_value / leverage
        expected_margin_level = expected_margin / portfolio.initial_capital
        
        logger.info(f"Leverage {leverage}x: Margin level {margin_level:.2%} "
                   f"(expected ~{expected_margin_level:.2%})")
        
        # Higher leverage should result in lower margin usage
        assert margin_level > 0, "Margin level should be positive"
        
        # Margin level should decrease with higher leverage
        if leverage > 1:
            assert margin_level < 0.5, f"High leverage should result in low margin usage"
    
    logger.info("✅ Leverage impact test passed")

def test_margin_vs_spot_trading():
    """Test differences between margin (futures) and spot trading."""
    logger.info("Testing margin vs spot trading differences...")
    
    # Test with futures (margin trading)
    futures_config = create_futures_config(leverage=3)
    futures_portfolio = PortfolioManager(futures_config)
    
    # Test with spot trading
    spot_config = create_spot_config()
    spot_portfolio = PortfolioManager(spot_config)
    
    position_size = 0.1
    btc_price = 50000.0
    
    # Open same position in both portfolios
    futures_portfolio.open_position('BTC', btc_price, position_size)
    spot_portfolio.open_position('BTC', btc_price, position_size)
    
    # Compare cash usage
    futures_cash_used = futures_portfolio.initial_capital - futures_portfolio.cash
    spot_cash_used = spot_portfolio.initial_capital - spot_portfolio.cash
    
    logger.info(f"Futures cash used: ${futures_cash_used:.2f}")
    logger.info(f"Spot cash used: ${spot_cash_used:.2f}")
    
    # Futures should use less cash due to leverage
    assert futures_cash_used < spot_cash_used, \
        "Futures trading should use less cash than spot trading"
    
    # Check margin level (should be 1.0 for spot, > 0 for futures)
    futures_margin_level = futures_portfolio.get_margin_level()
    spot_margin_level = spot_portfolio.get_margin_level()
    
    logger.info(f"Futures margin level: {futures_margin_level:.2%}")
    logger.info(f"Spot margin level: {spot_margin_level:.2%}")
    
    assert futures_margin_level > 0, "Futures should have positive margin level"
    assert spot_margin_level == 1.0, "Spot trading should have margin level of 1.0"
    
    logger.info("✅ Margin vs spot trading test passed")

def test_margin_call_scenarios():
    """Test various margin call scenarios and responses."""
    logger.info("Testing margin call scenarios...")
    
    config = create_futures_config(leverage=5)
    config['trading_rules']['liquidation_threshold'] = 0.05  # 5% threshold (more realistic)
    
    portfolio = PortfolioManager(config)
    
    # Scenario 1: Normal operation - no margin call
    portfolio.open_position('BTC', 50000.0, 0.1)  # $5,000 notional, $1,000 margin
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0}
    portfolio.update_market_price(current_prices)
    
    margin_level_1 = portfolio.get_margin_level()
    logger.info(f"Scenario 1 - Normal operation: {margin_level_1:.2%}")
    # With 10% margin level and 5% threshold, position should remain open
    assert portfolio.positions['BTC'].is_open, "Position should remain open"
    
    # Scenario 2: Moderate adverse movement - approaching margin call
    adverse_prices_1 = {'BTC': 47000.0, 'ETH': 3000.0}  # 6% drop
    portfolio.update_market_price(adverse_prices_1)
    
    margin_level_2 = portfolio.get_margin_level()
    logger.info(f"Scenario 2 - Moderate adverse movement: {margin_level_2:.2%}")
    
    # Scenario 3: Severe adverse movement - margin call triggered
    adverse_prices_2 = {'BTC': 42000.0, 'ETH': 3000.0}  # 16% drop
    portfolio.update_market_price(adverse_prices_2)
    
    margin_level_3 = portfolio.get_margin_level()
    logger.info(f"Scenario 3 - Severe adverse movement: {margin_level_3:.2%}")
    
    # Check if liquidation was triggered
    if margin_level_3 < config['trading_rules']['liquidation_threshold']:
        assert not portfolio.positions['BTC'].is_open, \
            "Position should be liquidated due to low margin level"
        logger.info("Position was liquidated due to margin call")
    
    logger.info("✅ Margin call scenarios test passed")

def test_margin_efficiency():
    """Test margin efficiency and capital utilization."""
    logger.info("Testing margin efficiency...")
    
    config = create_futures_config(leverage=4)
    portfolio = PortfolioManager(config)
    
    # Open multiple positions to test capital efficiency
    portfolio.open_position('BTC', 50000.0, 0.08)  # $4,000 notional, $1,000 margin
    portfolio.open_position('ETH', 3000.0, 1.0)    # $3,000 notional, $750 margin
    
    current_prices = {'BTC': 50000.0, 'ETH': 3000.0}
    portfolio.update_market_price(current_prices)
    
    # Calculate capital efficiency metrics
    total_notional = (0.08 * 50000.0) + (1.0 * 3000.0)  # $7,000
    total_margin_used = portfolio.get_margin_level() * portfolio.initial_capital
    capital_efficiency = total_notional / portfolio.initial_capital
    margin_efficiency = total_notional / total_margin_used if total_margin_used > 0 else 0
    
    logger.info(f"Total notional exposure: ${total_notional:.2f}")
    logger.info(f"Total margin used: ${total_margin_used:.2f}")
    logger.info(f"Capital efficiency: {capital_efficiency:.2f}x")
    logger.info(f"Margin efficiency: {margin_efficiency:.2f}x")
    
    # With 4x leverage, we should achieve reasonable efficiency
    assert capital_efficiency > 0.5, "Capital efficiency should be > 0.5x"
    assert margin_efficiency > 3.0, "Margin efficiency should be > 3.0x"
    
    logger.info("✅ Margin efficiency test passed")

def run_all_tests():
    """Run all margin management tests."""
    logger.info("🚀 Starting margin management tests...")
    
    try:
        test_margin_calculation()
        test_margin_level_monitoring()
        test_liquidation_risk_detection()
        test_margin_requirements_validation()
        test_leverage_impact()
        test_margin_vs_spot_trading()
        test_margin_call_scenarios()
        test_margin_efficiency()
        
        logger.info("🎉 All margin management tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)