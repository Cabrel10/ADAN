#!/usr/bin/env python3
"""
Test script for position validation functionality in PortfolioManager.

This script tests:
1. Basic position validation rules
2. Size and price validation
3. Notional value limits
4. Cash availability checks
5. Margin requirements validation
6. Risk-based validation
7. Edge cases and error handling
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

def create_test_config(futures_enabled: bool = False, leverage: int = 1) -> Dict[str, Any]:
    """Create a test configuration for the PortfolioManager."""
    return {
        'initial_balance': 10000.0,
        'assets': ['BTC', 'ETH', 'ADA'],
        'trading_rules': {
            'futures_enabled': futures_enabled,
            'leverage': leverage,
            'commission_pct': 0.001,
            'futures_commission_pct': 0.0005,
            'min_trade_size': 0.001,
            'min_notional_value': 10.0,
            'max_notional_value': 5000.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'liquidation_threshold': 0.1
        },
        'risk_management': {
            'capital_tiers': [
                {'threshold': 0, 'allocation_per_trade': 0.2}
            ],
            'position_sizing': {
                'concentration_limits': {
                    'max_single_asset': 0.4
                }
            }
        }
    }

def test_basic_validation_rules():
    """Test basic position validation rules."""
    logger.info("Testing basic validation rules...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test 1: Valid position
    assert portfolio.validate_position('BTC', 0.1, 50000.0), \
        "Valid position should pass validation"
    
    # Test 2: Invalid size (zero)
    assert not portfolio.validate_position('BTC', 0.0, 50000.0), \
        "Zero size should fail validation"
    
    # Test 3: Invalid size (negative)
    assert not portfolio.validate_position('BTC', -0.1, 50000.0), \
        "Negative size should fail validation"
    
    # Test 4: Invalid price (zero)
    assert not portfolio.validate_position('BTC', 0.1, 0.0), \
        "Zero price should fail validation"
    
    # Test 5: Invalid price (negative)
    assert not portfolio.validate_position('BTC', 0.1, -1000.0), \
        "Negative price should fail validation"
    
    logger.info("✅ Basic validation rules test passed")

def test_size_constraints():
    """Test position size constraints."""
    logger.info("Testing size constraints...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test minimum trade size
    min_size = config['trading_rules']['min_trade_size']
    
    # Test 1: Below minimum trade size
    assert not portfolio.validate_position('BTC', min_size / 2, 50000.0), \
        "Position below minimum trade size should fail validation"
    
    # Test 2: At minimum trade size
    assert portfolio.validate_position('BTC', min_size, 50000.0), \
        "Position at minimum trade size should pass validation"
    
    # Test 3: Above minimum trade size
    assert portfolio.validate_position('BTC', min_size * 2, 50000.0), \
        "Position above minimum trade size should pass validation"
    
    logger.info("✅ Size constraints test passed")

def test_notional_value_limits():
    """Test notional value limits validation."""
    logger.info("Testing notional value limits...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    min_notional = config['trading_rules']['min_notional_value']
    max_notional = config['trading_rules']['max_notional_value']
    
    # Test 1: Below minimum notional value
    low_price = min_notional / 2  # This will create notional value below minimum
    assert not portfolio.validate_position('BTC', 0.001, low_price), \
        "Position below minimum notional value should fail validation"
    
    # Test 2: At minimum notional value
    exact_price = min_notional / 0.001  # This will create exact minimum notional
    assert portfolio.validate_position('BTC', 0.001, exact_price), \
        "Position at minimum notional value should pass validation"
    
    # Test 3: Above maximum notional value
    high_size = max_notional / 1000 + 0.1  # This will exceed max notional at $1000/unit
    assert not portfolio.validate_position('BTC', high_size, 1000.0), \
        "Position above maximum notional value should fail validation"
    
    # Test 4: At maximum notional value
    exact_size = max_notional / 1000  # This will create exact maximum notional
    assert portfolio.validate_position('BTC', exact_size, 1000.0), \
        "Position at maximum notional value should pass validation"
    
    logger.info("✅ Notional value limits test passed")

def test_cash_availability():
    """Test cash availability validation."""
    logger.info("Testing cash availability...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test 1: Sufficient cash
    assert portfolio.validate_position('BTC', 0.1, 50000.0), \
        "Position with sufficient cash should pass validation"
    
    # Test 2: Insufficient cash
    portfolio.cash = 1000.0  # Reduce available cash
    assert not portfolio.validate_position('BTC', 0.1, 50000.0), \
        "Position with insufficient cash should fail validation"
    
    # Test 3: Exactly enough cash
    portfolio.cash = 5005.0  # Exactly enough for position + commission
    assert portfolio.validate_position('BTC', 0.1, 50000.0), \
        "Position with exactly enough cash should pass validation"
    
    logger.info("✅ Cash availability test passed")

def test_margin_requirements_validation():
    """Test margin requirements validation for futures."""
    logger.info("Testing margin requirements validation...")
    
    config = create_test_config(futures_enabled=True, leverage=5)
    portfolio = PortfolioManager(config)
    
    # Test 1: Valid margin position
    assert portfolio.validate_position('BTC', 0.1, 50000.0), \
        "Valid margin position should pass validation"
    
    # Test 2: Position requiring too much margin
    portfolio.cash = 500.0  # Reduce available cash
    # Position requires $1000 margin + commission, but only $500 available
    assert not portfolio.validate_position('BTC', 0.1, 50000.0), \
        "Position requiring excessive margin should fail validation"
    
    # Test 3: Position with acceptable margin
    portfolio.cash = 2000.0  # Sufficient for margin
    assert portfolio.validate_position('BTC', 0.05, 50000.0), \
        "Position with acceptable margin should pass validation"
    
    logger.info("✅ Margin requirements validation test passed")

def test_leverage_impact_on_validation():
    """Test how different leverage levels affect validation."""
    logger.info("Testing leverage impact on validation...")
    
    leverages = [1, 3, 5, 10]
    position_size = 0.1
    btc_price = 50000.0
    
    for leverage in leverages:
        config = create_test_config(futures_enabled=True, leverage=leverage)
        portfolio = PortfolioManager(config)
        
        # Set cash to a level that should work with higher leverage but not lower
        portfolio.cash = 1500.0
        
        result = portfolio.validate_position('BTC', position_size, btc_price)
        
        # Calculate expected margin requirement
        notional_value = position_size * btc_price
        margin_required = notional_value / leverage
        commission = notional_value * config['trading_rules']['futures_commission_pct']
        total_required = margin_required + commission
        
        logger.info(f"Leverage {leverage}x: Required ${total_required:.2f}, "
                   f"Available ${portfolio.cash:.2f}, Valid: {result}")
        
        # Validation should pass if we have enough cash for margin + commission
        expected_result = portfolio.cash >= total_required
        assert result == expected_result, \
            f"Validation result should be {expected_result} for leverage {leverage}x"
    
    logger.info("✅ Leverage impact on validation test passed")

def test_risk_based_validation():
    """Test risk-based validation rules."""
    logger.info("Testing risk-based validation...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test 1: Position within risk limits
    assert portfolio.validate_position('BTC', 0.05, 50000.0), \
        "Position within risk limits should pass validation"
    
    # Test 2: Very large position (risk assessment)
    # This should still pass basic validation but might be flagged by risk management
    large_size = 0.5  # Large position
    result = portfolio.validate_position('BTC', large_size, 1000.0)
    
    # The validation should consider the notional value limits
    notional_value = large_size * 1000.0
    max_notional = config['trading_rules']['max_notional_value']
    
    if notional_value > max_notional:
        assert not result, "Position exceeding notional limits should fail"
    else:
        assert result, "Position within notional limits should pass"
    
    logger.info("✅ Risk-based validation test passed")

def test_validation_with_existing_positions():
    """Test validation when there are existing positions."""
    logger.info("Testing validation with existing positions...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Open an existing position
    portfolio.open_position('BTC', 40000.0, 0.1)
    
    # Test 1: Valid additional position
    assert portfolio.validate_position('ETH', 1.0, 3000.0), \
        "Valid additional position should pass validation"
    
    # Test 2: Position that would exceed available cash
    # After opening BTC position, less cash is available
    remaining_cash = portfolio.cash
    large_eth_size = (remaining_cash / 3000.0) + 0.1  # Slightly more than available
    
    assert not portfolio.validate_position('ETH', large_eth_size, 3000.0), \
        "Position exceeding remaining cash should fail validation"
    
    # Test 3: Position for asset that already has an open position
    # This should still validate the position parameters
    assert portfolio.validate_position('BTC', 0.05, 45000.0), \
        "Validation should work even for assets with existing positions"
    
    logger.info("✅ Validation with existing positions test passed")

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    logger.info("Testing edge cases...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test 1: Very small position
    tiny_size = 0.00000001
    assert not portfolio.validate_position('BTC', tiny_size, 50000.0), \
        "Extremely small position should fail validation"
    
    # Test 2: Very large price
    huge_price = 1000000.0
    small_size = 0.001
    # This might fail due to notional value limits
    result = portfolio.validate_position('BTC', small_size, huge_price)
    notional = small_size * huge_price
    expected = notional <= config['trading_rules']['max_notional_value']
    assert result == expected, "Validation should respect notional value limits"
    
    # Test 3: Boundary values
    min_size = config['trading_rules']['min_trade_size']
    min_notional = config['trading_rules']['min_notional_value']
    
    # Exact minimum size
    assert portfolio.validate_position('BTC', min_size, 50000.0), \
        "Exact minimum size should pass validation"
    
    # Price that creates exact minimum notional
    exact_price = min_notional / min_size
    assert portfolio.validate_position('BTC', min_size, exact_price), \
        "Position creating exact minimum notional should pass validation"
    
    logger.info("✅ Edge cases test passed")

def test_validation_error_messages():
    """Test that validation provides appropriate error context."""
    logger.info("Testing validation error messages...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    # Test various invalid scenarios and ensure they fail
    # (The actual error messages are logged, not returned)
    
    test_cases = [
        (0.0, 50000.0, "zero size"),
        (-0.1, 50000.0, "negative size"),
        (0.1, 0.0, "zero price"),
        (0.1, -1000.0, "negative price"),
        (0.0001, 50000.0, "below minimum trade size"),
        (0.0001, 1.0, "below minimum notional value"),
        (10.0, 1000.0, "above maximum notional value")
    ]
    
    for size, price, description in test_cases:
        result = portfolio.validate_position('BTC', size, price)
        assert not result, f"Validation should fail for {description}"
        logger.info(f"✓ Correctly rejected position with {description}")
    
    # Test insufficient cash scenario
    portfolio.cash = 100.0
    result = portfolio.validate_position('BTC', 0.1, 50000.0)
    assert not result, "Validation should fail for insufficient cash"
    logger.info("✓ Correctly rejected position with insufficient cash")
    
    logger.info("✅ Validation error messages test passed")

def test_validation_performance():
    """Test validation performance with multiple calls."""
    logger.info("Testing validation performance...")
    
    config = create_test_config()
    portfolio = PortfolioManager(config)
    
    import time
    
    # Test validation performance
    start_time = time.time()
    
    for i in range(1000):
        portfolio.validate_position('BTC', 0.1, 50000.0 + i)
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"1000 validations completed in {duration:.4f} seconds")
    logger.info(f"Average validation time: {duration/1000*1000:.4f} ms")
    
    # Validation should be fast (less than 1ms per validation on average)
    assert duration < 1.0, "Validation should be fast"
    
    logger.info("✅ Validation performance test passed")

def run_all_tests():
    """Run all position validation tests."""
    logger.info("🚀 Starting position validation tests...")
    
    try:
        test_basic_validation_rules()
        test_size_constraints()
        test_notional_value_limits()
        test_cash_availability()
        test_margin_requirements_validation()
        test_leverage_impact_on_validation()
        test_risk_based_validation()
        test_validation_with_existing_positions()
        test_edge_cases()
        test_validation_error_messages()
        test_validation_performance()
        
        logger.info("🎉 All position validation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)