#!/usr/bin/env python3
"""
Test script for action validation in the ADAN Trading Bot.

This script tests the action validation functionality to ensure that
trading actions are properly validated against portfolio and market constraints.
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.trading.action_translator import (
    ActionTranslator, ActionType, PositionSizeMethod, TradingAction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_action_validation():
    """Test basic action validation functionality."""
    logger.info("Testing basic action validation...")
    
    # Initialize ActionTranslator
    translator = ActionTranslator(
        action_space_type="discrete",
        position_size_method=PositionSizeMethod.PERCENTAGE,
        default_position_size=0.1,
        max_position_size=0.5,
        min_position_size=0.01
    )
    
    # Test portfolio state
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {
            'BTC': 0.5,  # Long position
            'ETH': -0.2  # Short position
        }
    }
    
    # Test valid BUY action
    buy_action = TradingAction(
        action_type=ActionType.BUY,
        asset='BTC',
        size=0.1,
        price=50000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        buy_action, portfolio_state
    )
    
    assert validation_result.is_valid, f"Valid BUY action failed validation: {validation_result.message}"
    logger.info("✅ Valid BUY action passed validation")
    
    # Test invalid action - negative size
    invalid_action = TradingAction(
        action_type=ActionType.BUY,
        asset='BTC',
        size=-0.1,
        price=50000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        invalid_action, portfolio_state
    )
    
    assert not validation_result.is_valid, "Invalid action with negative size should fail validation"
    logger.info("✅ Invalid action with negative size correctly rejected")
    
    return True


def test_capital_constraints_validation():
    """Test validation of capital constraints."""
    logger.info("Testing capital constraints validation...")
    
    translator = ActionTranslator()
    
    # Portfolio with limited cash
    portfolio_state = {
        'cash': 1000.0,
        'total_value': 1000.0,
        'positions': {}
    }
    
    # Action requiring more capital than available
    expensive_action = TradingAction(
        action_type=ActionType.BUY,
        asset='BTC',
        size=1.0,  # 1 BTC at 50000 = 50000 USD needed
        price=50000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        expensive_action, portfolio_state
    )
    
    # Should either be invalid or have adjusted action
    if validation_result.is_valid:
        assert validation_result.adjusted_action is not None, "Should have adjusted action for capital constraints"
        assert validation_result.adjusted_action.size < expensive_action.size, "Adjusted size should be smaller"
        logger.info("✅ Capital constraints handled with position size adjustment")
    else:
        logger.info("✅ Capital constraints correctly rejected expensive action")
    
    return True


def test_position_constraints_validation():
    """Test validation of position constraints."""
    logger.info("Testing position constraints validation...")
    
    translator = ActionTranslator()
    
    # Portfolio with existing positions
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {
            'BTC': 0.5,   # Long position
            'ETH': -0.2,  # Short position
            'ADA': 0.0    # No position
        }
    }
    
    # Test closing long position that exists
    close_long_action = TradingAction(
        action_type=ActionType.CLOSE_LONG,
        asset='BTC',
        size=0.5,
        price=50000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        close_long_action, portfolio_state
    )
    
    assert validation_result.is_valid, f"Valid CLOSE_LONG action failed: {validation_result.message}"
    logger.info("✅ Valid CLOSE_LONG action passed validation")
    
    # Test closing long position that doesn't exist
    invalid_close_long = TradingAction(
        action_type=ActionType.CLOSE_LONG,
        asset='ADA',  # No position
        size=0.1,
        price=1.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        invalid_close_long, portfolio_state
    )
    
    assert not validation_result.is_valid, "Should not be able to close non-existent long position"
    logger.info("✅ Invalid CLOSE_LONG action correctly rejected")
    
    # Test closing short position that exists
    close_short_action = TradingAction(
        action_type=ActionType.CLOSE_SHORT,
        asset='ETH',
        size=0.2,
        price=3000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        close_short_action, portfolio_state
    )
    
    assert validation_result.is_valid, f"Valid CLOSE_SHORT action failed: {validation_result.message}"
    logger.info("✅ Valid CLOSE_SHORT action passed validation")
    
    return True


def test_market_constraints_validation():
    """Test validation against market constraints."""
    logger.info("Testing market constraints validation...")
    
    translator = ActionTranslator()
    
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {}
    }
    
    # Market constraints
    market_constraints = {
        'min_order_size': 0.01,
        'max_order_size': 10.0,
        'tick_size': 0.01
    }
    
    # Test action below minimum size
    small_action = TradingAction(
        action_type=ActionType.BUY,
        asset='BTC',
        size=0.005,  # Below minimum
        price=50000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        small_action, portfolio_state, market_constraints
    )
    
    assert not validation_result.is_valid, "Action below minimum size should be rejected"
    logger.info("✅ Action below minimum size correctly rejected")
    
    # Test action above maximum size
    large_action = TradingAction(
        action_type=ActionType.BUY,
        asset='BTC',
        size=15.0,  # Above maximum
        price=50000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        large_action, portfolio_state, market_constraints
    )
    
    # Should either be invalid or have adjusted action
    if validation_result.is_valid:
        assert validation_result.adjusted_action is not None, "Should have adjusted action for size constraints"
        assert validation_result.adjusted_action.size <= market_constraints['max_order_size'], "Adjusted size should be within limits"
        logger.info("✅ Large action handled with size adjustment")
    else:
        logger.info("✅ Large action correctly rejected")
    
    return True


def test_stop_loss_take_profit_validation():
    """Test validation of stop loss and take profit levels."""
    logger.info("Testing stop loss and take profit validation...")
    
    translator = ActionTranslator()
    
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {}
    }
    
    # Test BUY action with invalid stop loss (above entry price)
    buy_action_invalid_sl = TradingAction(
        action_type=ActionType.BUY,
        asset='BTC',
        size=0.1,
        price=50000.0,
        stop_loss=55000.0,  # Above entry price - invalid for buy
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        buy_action_invalid_sl, portfolio_state
    )
    
    # Should be valid but with warnings
    assert validation_result.is_valid, "Action should be valid despite stop loss warning"
    assert validation_result.warnings, "Should have warnings about stop loss level"
    logger.info("✅ Invalid stop loss level generated appropriate warning")
    
    # Test SELL action with invalid stop loss (below entry price)
    sell_action_invalid_sl = TradingAction(
        action_type=ActionType.SELL,
        asset='BTC',
        size=0.1,
        price=50000.0,
        stop_loss=45000.0,  # Below entry price - invalid for sell
        confidence=0.8
    )
    
    validation_result = translator.validate_action(
        sell_action_invalid_sl, portfolio_state
    )
    
    # Should be valid but with warnings
    assert validation_result.is_valid, "Action should be valid despite stop loss warning"
    assert validation_result.warnings, "Should have warnings about stop loss level"
    logger.info("✅ Invalid stop loss level for sell generated appropriate warning")
    
    return True


def test_validation_error_handling():
    """Test error handling in validation."""
    logger.info("Testing validation error handling...")
    
    translator = ActionTranslator()
    
    # Test with invalid portfolio state
    invalid_portfolio = None
    
    action = TradingAction(
        action_type=ActionType.BUY,
        asset='BTC',
        size=0.1,
        price=50000.0,
        confidence=0.8
    )
    
    try:
        validation_result = translator.validate_action(action, invalid_portfolio)
        # Should handle error gracefully
        assert not validation_result.is_valid, "Should handle invalid portfolio gracefully"
        logger.info("✅ Invalid portfolio state handled gracefully")
    except Exception as e:
        logger.error(f"Validation should handle errors gracefully, but got: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all action validation tests."""
    logger.info("Starting action validation tests...")
    
    tests = [
        test_basic_action_validation,
        test_capital_constraints_validation,
        test_position_constraints_validation,
        test_market_constraints_validation,
        test_stop_loss_take_profit_validation,
        test_validation_error_handling
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
        logger.info("🎉 All action validation tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)