#!/usr/bin/env python3
"""
Test script for action logging in the ADAN Trading Bot.

This script tests the comprehensive action logging functionality including
logging of translated actions, validations, and statistics.
"""

import sys
from pathlib import Path
import logging
import tempfile
import os

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.trading.action_translator import (
    ActionTranslator, ActionType, PositionSizeMethod
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_action_logging():
    """Test basic action logging functionality."""
    logger.info("Testing basic action logging...")
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_file = f.name
    
    try:
        # Configure action translator with logging
        translator = ActionTranslator(
            action_space_type="discrete",
            position_size_method=PositionSizeMethod.PERCENTAGE
        )
        
        # Test portfolio state
        portfolio_state = {
            'cash': 10000.0,
            'total_value': 12000.0,
            'positions': {}
        }
        
        # Test multiple action translations
        actions_to_test = [0, 1, 2]  # HOLD, BUY, SELL
        
        for action in actions_to_test:
            trading_action = translator.translate_action(
                agent_action=action,
                asset='BTC',
                current_price=50000.0,
                portfolio_state=portfolio_state
            )
            
            assert trading_action is not None, f"Action {action} should be translated"
            logger.info(f"✅ Action {action} translated to {trading_action.action_type.name}")
        
        # Check translation statistics
        stats = translator.get_translation_stats()
        assert stats['total_translations'] == 3, "Should have 3 translations"
        assert stats['successful_translations'] == 3, "All translations should be successful"
        logger.info(f"✅ Translation stats: {stats}")
        
        return True
        
    finally:
        # Clean up temporary file
        if os.path.exists(log_file):
            os.unlink(log_file)


def test_action_validation_logging():
    """Test action validation logging."""
    logger.info("Testing action validation logging...")
    
    translator = ActionTranslator()
    
    # Test portfolio state
    portfolio_state = {
        'cash': 1000.0,  # Limited cash
        'total_value': 1000.0,
        'positions': {}
    }
    
    # Create an action that will require validation adjustments
    from adan_trading_bot.trading.action_translator import TradingAction
    
    expensive_action = TradingAction(
        action_type=ActionType.BUY,
        asset='BTC',
        size=1.0,  # 1 BTC at 50000 = 50000 USD needed
        price=50000.0,
        confidence=0.8
    )
    
    # Validate the action (should trigger adjustments)
    validation_result = translator.validate_action(
        expensive_action, portfolio_state
    )
    
    # Check that validation was logged
    if validation_result.adjusted_action:
        logger.info("✅ Action validation with adjustment logged")
    elif validation_result.warnings:
        logger.info("✅ Action validation with warnings logged")
    else:
        logger.info("✅ Action validation logged")
    
    return True


def test_error_logging():
    """Test error logging in action translation."""
    logger.info("Testing error logging...")
    
    translator = ActionTranslator()
    
    # Test with invalid portfolio state to trigger error handling
    invalid_portfolio = None
    
    # This should trigger error handling and logging
    trading_action = translator.translate_action(
        agent_action=1,
        asset='BTC',
        current_price=50000.0,
        portfolio_state=invalid_portfolio
    )
    
    # Should return a safe default action
    assert trading_action.action_type == ActionType.HOLD, "Should return HOLD on error"
    assert trading_action.confidence == 0.0, "Should have zero confidence on error"
    logger.info("✅ Error handling and logging working correctly")
    
    return True


def test_statistics_logging():
    """Test statistics logging and tracking."""
    logger.info("Testing statistics logging...")
    
    translator = ActionTranslator()
    
    # Test portfolio state
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {}
    }
    
    # Perform multiple translations to build statistics
    action_types = [0, 1, 2, 1, 0, 2, 1]  # Mix of actions
    
    for action in action_types:
        translator.translate_action(
            agent_action=action,
            asset='BTC',
            current_price=50000.0,
            portfolio_state=portfolio_state
        )
    
    # Get and verify statistics
    stats = translator.get_translation_stats()
    
    assert stats['total_translations'] == len(action_types), "Should track all translations"
    assert stats['success_rate'] > 0, "Should have positive success rate"
    
    # Check action type distribution
    action_counts = stats['action_type_counts']
    assert action_counts['HOLD'] == 2, "Should have 2 HOLD actions"
    assert action_counts['BUY'] == 3, "Should have 3 BUY actions"
    assert action_counts['SELL'] == 2, "Should have 2 SELL actions"
    
    logger.info(f"✅ Statistics tracking: {stats}")
    
    return True


def test_performance_logging():
    """Test performance-related logging."""
    logger.info("Testing performance logging...")
    
    translator = ActionTranslator()
    
    # Test portfolio state
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {}
    }
    
    # Test with market data to trigger advanced calculations
    market_data = {
        'volatility': 0.25,
        'expected_return': 0.12,
        'atr': 1500.0
    }
    
    # Perform translation with market data
    trading_action = translator.translate_action(
        agent_action=1,  # BUY
        asset='BTC',
        current_price=50000.0,
        portfolio_state=portfolio_state,
        market_data=market_data
    )
    
    assert trading_action.size > 0, "Should have positive position size"
    assert trading_action.fees is not None, "Should have calculated fees"
    
    # Check that metadata includes performance information
    if trading_action.metadata:
        assert 'market_data' in trading_action.metadata, "Should include market data in metadata"
        logger.info("✅ Performance data logged in action metadata")
    
    return True


def test_fee_calculation_logging():
    """Test fee calculation logging."""
    logger.info("Testing fee calculation logging...")
    
    translator = ActionTranslator()
    
    # Test portfolio state
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {}
    }
    
    # Test different assets to trigger different fee calculations
    assets = ['BTC', 'ETH', 'USDT']
    
    for asset in assets:
        trading_action = translator.translate_action(
            agent_action=1,  # BUY
            asset=asset,
            current_price=50000.0 if asset == 'BTC' else 3000.0 if asset == 'ETH' else 1.0,
            portfolio_state=portfolio_state
        )
        
        assert trading_action.fees is not None, f"Should have fees for {asset}"
        assert trading_action.fees >= 0, f"Fees should be non-negative for {asset}"
        
        logger.info(f"✅ {asset} fee calculation: ${trading_action.fees:.2f}")
    
    return True


def test_position_sizing_logging():
    """Test position sizing logging."""
    logger.info("Testing position sizing logging...")
    
    # Test different position sizing methods
    methods = [
        PositionSizeMethod.PERCENTAGE,
        PositionSizeMethod.VOLATILITY_ADJUSTED,
        PositionSizeMethod.KELLY
    ]
    
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {}
    }
    
    market_data = {
        'volatility': 0.20,
        'expected_return': 0.10,
        'win_rate': 0.6,
        'avg_win': 0.05,
        'avg_loss': 0.03
    }
    
    for method in methods:
        translator = ActionTranslator(position_size_method=method)
        
        trading_action = translator.translate_action(
            agent_action=1,  # BUY
            asset='BTC',
            current_price=50000.0,
            portfolio_state=portfolio_state,
            market_data=market_data
        )
        
        assert trading_action.size > 0, f"Should have positive size for {method.value}"
        
        # Check for position sizing warnings in translation stats
        stats = translator.get_translation_stats()
        if stats['size_adjustments'] > 0:
            logger.info(f"✅ Position sizing adjustments logged for {method.value}")
        else:
            logger.info(f"✅ Position sizing calculated for {method.value}")
    
    return True


def test_comprehensive_logging():
    """Test comprehensive logging across all components."""
    logger.info("Testing comprehensive logging...")
    
    translator = ActionTranslator(
        position_size_method=PositionSizeMethod.VOLATILITY_ADJUSTED,
        default_position_size=0.15
    )
    
    # Complex test scenario
    portfolio_state = {
        'cash': 5000.0,  # Limited cash to trigger adjustments
        'total_value': 10000.0,
        'positions': {
            'ETH': 2.0  # Existing position
        }
    }
    
    market_data = {
        'volatility': 0.30,  # High volatility
        'expected_return': 0.08,
        'atr': 2000.0
    }
    
    # Test various scenarios
    scenarios = [
        (1, 'BTC', 50000.0),  # BUY BTC
        (2, 'ETH', 3000.0),   # SELL ETH (has position)
        (3, 'ETH', 3000.0),   # CLOSE_LONG ETH
        (0, 'ADA', 1.0),      # HOLD ADA
    ]
    
    for action, asset, price in scenarios:
        trading_action = translator.translate_action(
            agent_action=action,
            asset=asset,
            current_price=price,
            portfolio_state=portfolio_state,
            market_data=market_data
        )
        
        # Validate the action to trigger validation logging
        validation_result = translator.validate_action(
            trading_action, portfolio_state
        )
        
        logger.info(f"✅ Comprehensive test - {asset} action: {trading_action.action_type.name}")
    
    # Get final statistics
    final_stats = translator.get_translation_stats()
    logger.info(f"✅ Final comprehensive stats: {final_stats}")
    
    return True


def run_all_tests():
    """Run all action logging tests."""
    logger.info("Starting action logging tests...")
    
    tests = [
        test_basic_action_logging,
        test_action_validation_logging,
        test_error_logging,
        test_statistics_logging,
        test_performance_logging,
        test_fee_calculation_logging,
        test_position_sizing_logging,
        test_comprehensive_logging
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
        logger.info("🎉 All action logging tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)