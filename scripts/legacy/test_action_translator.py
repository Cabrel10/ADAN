#!/usr/bin/env python3
"""
Test script for ActionTranslator functionality.

This script tests the comprehensive action translation system to ensure
proper conversion from RL agent actions to executable trading actions.
"""

import numpy as np
from pathlib import Path
import sys
import logging
import traceback

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.trading.action_translator import (
    ActionTranslator, ActionType, PositionSizeMethod, TradingAction
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_portfolio_state():
    """Create a test portfolio state."""
    return {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {
            'BTC': 0.5,  # Long position
            'ETH': -0.3,  # Short position
            'ADA': 0.0   # No position
        }
    }

def create_test_market_data():
    """Create test market data."""
    return {
        'volatility': 0.03,
        'expected_return': 0.05,
        'volume': 1000000,
        'bid_ask_spread': 0.001
    }

def test_discrete_action_translation():
    """Test discrete action translation."""
    logger.info("🔍 Testing discrete action translation...")
    
    translator = ActionTranslator(
        action_space_type="discrete",
        position_size_method=PositionSizeMethod.PERCENTAGE,
        default_position_size=0.1
    )
    
    portfolio_state = create_test_portfolio_state()
    market_data = create_test_market_data()
    
    # Test each discrete action
    test_cases = [
        (0, ActionType.HOLD, "Hold action"),
        (1, ActionType.BUY, "Buy action"),
        (2, ActionType.SELL, "Sell action"),
        (3, ActionType.CLOSE_LONG, "Close long action"),
        (4, ActionType.CLOSE_SHORT, "Close short action")
    ]
    
    for action_idx, expected_type, description in test_cases:
        logger.info(f"📊 Testing {description}...")
        
        trading_action = translator.translate_action(
            agent_action=action_idx,
            asset="BTC",
            current_price=50000.0,
            portfolio_state=portfolio_state,
            market_data=market_data
        )
        
        assert trading_action.action_type == expected_type, f"Expected {expected_type}, got {trading_action.action_type}"
        assert trading_action.asset == "BTC", "Asset should be BTC"
        assert trading_action.price == 50000.0, "Price should match current price"
        
        logger.info(f"   Action: {trading_action.action_type.name}, Size: {trading_action.size:.4f}")
        
        if expected_type != ActionType.HOLD:
            assert trading_action.size > 0, "Non-hold actions should have positive size"
        else:
            assert trading_action.size == 0, "Hold action should have zero size"
    
    logger.info("   ✅ Discrete action translation test passed")
    return True

def test_continuous_action_translation():
    """Test continuous action translation."""
    logger.info("🔄 Testing continuous action translation...")
    
    translator = ActionTranslator(
        action_space_type="continuous",
        position_size_method=PositionSizeMethod.VOLATILITY_ADJUSTED
    )
    
    portfolio_state = create_test_portfolio_state()
    market_data = create_test_market_data()
    
    # Test continuous action cases
    test_cases = [
        ([0.0, 0.5], ActionType.HOLD, "Neutral action"),
        ([0.8, 0.7], ActionType.BUY, "Strong buy action"),
        ([-0.8, 0.6], ActionType.SELL, "Strong sell action"),
        ([0.3, 0.4], ActionType.CLOSE_SHORT, "Weak close short"),
        ([-0.3, 0.3], ActionType.CLOSE_LONG, "Weak close long")
    ]
    
    for action_values, expected_type, description in test_cases:
        logger.info(f"📈 Testing {description}...")
        
        trading_action = translator.translate_action(
            agent_action=np.array(action_values),
            asset="ETH",
            current_price=3000.0,
            portfolio_state=portfolio_state,
            market_data=market_data
        )
        
        assert trading_action.action_type == expected_type, f"Expected {expected_type}, got {trading_action.action_type}"
        assert 0.0 <= trading_action.confidence <= 1.0, "Confidence should be between 0 and 1"
        
        logger.info(f"   Action: {trading_action.action_type.name}, Size: {trading_action.size:.4f}, Confidence: {trading_action.confidence:.2f}")
    
    logger.info("   ✅ Continuous action translation test passed")
    return True

def test_position_sizing_methods():
    """Test different position sizing methods."""
    logger.info("💰 Testing position sizing methods...")
    
    portfolio_state = create_test_portfolio_state()
    market_data = create_test_market_data()
    
    sizing_methods = [
        (PositionSizeMethod.FIXED, "Fixed sizing"),
        (PositionSizeMethod.PERCENTAGE, "Percentage sizing"),
        (PositionSizeMethod.VOLATILITY_ADJUSTED, "Volatility adjusted sizing"),
        (PositionSizeMethod.KELLY, "Kelly criterion sizing")
    ]
    
    for method, description in sizing_methods:
        logger.info(f"📊 Testing {description}...")
        
        translator = ActionTranslator(
            position_size_method=method,
            default_position_size=0.1
        )
        
        trading_action = translator.translate_action(
            agent_action=1,  # BUY action
            asset="ADA",
            current_price=1.0,
            portfolio_state=portfolio_state,
            market_data=market_data
        )
        
        assert trading_action.size > 0, f"Buy action should have positive size for {method}"
        logger.info(f"   Method: {method.value}, Size: {trading_action.size:.4f}")
    
    logger.info("   ✅ Position sizing methods test passed")
    return True

def test_action_validation():
    """Test action validation functionality."""
    logger.info("✅ Testing action validation...")
    
    translator = ActionTranslator()
    portfolio_state = create_test_portfolio_state()
    
    # Test valid action
    logger.info("📋 Testing valid action validation...")
    valid_action = TradingAction(
        action_type=ActionType.BUY,
        asset="BTC",
        size=0.1,
        price=50000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(valid_action, portfolio_state)
    assert validation_result.is_valid, "Valid action should pass validation"
    logger.info("   ✅ Valid action validation passed")
    
    # Test insufficient capital
    logger.info("💸 Testing insufficient capital validation...")
    expensive_action = TradingAction(
        action_type=ActionType.BUY,
        asset="BTC",
        size=1.0,  # Requires 50,000 but only have 10,000 cash
        price=50000.0,
        confidence=0.8
    )
    
    validation_result = translator.validate_action(expensive_action, portfolio_state)
    # Should either be invalid or have adjusted action
    if not validation_result.is_valid:
        logger.info("   Action correctly rejected due to insufficient capital")
    elif validation_result.adjusted_action:
        logger.info(f"   Action adjusted: size reduced to {validation_result.adjusted_action.size:.4f}")
    else:
        assert False, "Should either reject or adjust expensive action"
    
    # Test invalid closing action
    logger.info("🚫 Testing invalid closing action...")
    invalid_close = TradingAction(
        action_type=ActionType.CLOSE_LONG,
        asset="ADA",  # No position in ADA
        size=0.1,
        price=1.0,
        confidence=0.5
    )
    
    validation_result = translator.validate_action(invalid_close, portfolio_state)
    assert not validation_result.is_valid, "Should reject closing non-existent position"
    logger.info("   ✅ Invalid closing action correctly rejected")
    
    logger.info("   ✅ Action validation test passed")
    return True

def test_stop_loss_take_profit():
    """Test stop loss and take profit calculation."""
    logger.info("🛡️ Testing stop loss and take profit calculation...")
    
    translator = ActionTranslator(
        enable_stop_loss=True,
        enable_take_profit=True,
        default_stop_loss_pct=0.02,
        default_take_profit_pct=0.04
    )
    
    portfolio_state = create_test_portfolio_state()
    market_data = create_test_market_data()
    
    # Test buy action
    logger.info("📈 Testing buy action stop/take levels...")
    buy_action = translator.translate_action(
        agent_action=1,  # BUY
        asset="BTC",
        current_price=50000.0,
        portfolio_state=portfolio_state,
        market_data=market_data
    )
    
    assert buy_action.stop_loss is not None, "Buy action should have stop loss"
    assert buy_action.take_profit is not None, "Buy action should have take profit"
    assert buy_action.stop_loss < 50000.0, "Stop loss should be below entry price for buy"
    assert buy_action.take_profit > 50000.0, "Take profit should be above entry price for buy"
    
    logger.info(f"   Buy - Entry: {buy_action.price}, SL: {buy_action.stop_loss:.2f}, TP: {buy_action.take_profit:.2f}")
    
    # Test sell action
    logger.info("📉 Testing sell action stop/take levels...")
    sell_action = translator.translate_action(
        agent_action=2,  # SELL
        asset="BTC",
        current_price=50000.0,
        portfolio_state=portfolio_state,
        market_data=market_data
    )
    
    assert sell_action.stop_loss is not None, "Sell action should have stop loss"
    assert sell_action.take_profit is not None, "Sell action should have take profit"
    assert sell_action.stop_loss > 50000.0, "Stop loss should be above entry price for sell"
    assert sell_action.take_profit < 50000.0, "Take profit should be below entry price for sell"
    
    logger.info(f"   Sell - Entry: {sell_action.price}, SL: {sell_action.stop_loss:.2f}, TP: {sell_action.take_profit:.2f}")
    
    logger.info("   ✅ Stop loss and take profit test passed")
    return True

def test_statistics_tracking():
    """Test statistics tracking functionality."""
    logger.info("📊 Testing statistics tracking...")
    
    translator = ActionTranslator()
    portfolio_state = create_test_portfolio_state()
    
    # Perform multiple translations
    for i in range(10):
        action_idx = i % 5  # Cycle through all action types
        translator.translate_action(
            agent_action=action_idx,
            asset="BTC",
            current_price=50000.0,
            portfolio_state=portfolio_state
        )
    
    # Get statistics
    stats = translator.get_translation_stats()
    
    assert stats['total_translations'] == 10, "Should have 10 total translations"
    assert stats['successful_translations'] == 10, "All translations should be successful"
    assert stats['success_rate'] == 100.0, "Success rate should be 100%"
    
    # Check action type counts
    action_counts = stats['action_type_counts']
    assert sum(action_counts.values()) == 10, "Action type counts should sum to total"
    
    logger.info(f"   Statistics: {stats}")
    logger.info("   ✅ Statistics tracking test passed")
    
    return True

def test_error_handling():
    """Test error handling in action translation."""
    logger.info("🚨 Testing error handling...")
    
    translator = ActionTranslator()
    portfolio_state = create_test_portfolio_state()
    
    # Test invalid action index
    logger.info("❌ Testing invalid action handling...")
    invalid_action = translator.translate_action(
        agent_action=999,  # Invalid action index
        asset="BTC",
        current_price=50000.0,
        portfolio_state=portfolio_state
    )
    
    # Should default to HOLD
    assert invalid_action.action_type == ActionType.HOLD, "Invalid action should default to HOLD"
    assert invalid_action.confidence == 0.0, "Invalid action should have zero confidence"
    
    # Test with invalid price
    logger.info("💰 Testing invalid price handling...")
    zero_price_action = translator.translate_action(
        agent_action=1,
        asset="BTC",
        current_price=0.0,  # Invalid price
        portfolio_state=portfolio_state
    )
    
    # Should still create action but handle gracefully
    assert zero_price_action is not None, "Should handle invalid price gracefully"
    
    logger.info("   ✅ Error handling test passed")
    return True

def test_configuration_updates():
    """Test configuration updates."""
    logger.info("⚙️ Testing configuration updates...")
    
    translator = ActionTranslator(default_position_size=0.1)
    
    # Update configuration
    translator.update_config(
        default_position_size=0.2,
        max_position_size=0.8,
        enable_stop_loss=False
    )
    
    assert translator.default_position_size == 0.2, "Should update default position size"
    assert translator.max_position_size == 0.8, "Should update max position size"
    assert translator.enable_stop_loss == False, "Should update stop loss setting"
    
    # Test action space info
    action_space_info = translator.get_action_space_info()
    assert 'type' in action_space_info, "Should provide action space type"
    assert action_space_info['type'] == 'discrete', "Should be discrete by default"
    
    logger.info("   ✅ Configuration updates test passed")
    return True

def main():
    """Main test function."""
    logger.info("🚀 Starting ActionTranslator tests...")
    
    try:
        # Test 1: Discrete action translation
        success1 = test_discrete_action_translation()
        
        # Test 2: Continuous action translation
        success2 = test_continuous_action_translation()
        
        # Test 3: Position sizing methods
        success3 = test_position_sizing_methods()
        
        # Test 4: Action validation
        success4 = test_action_validation()
        
        # Test 5: Stop loss and take profit
        success5 = test_stop_loss_take_profit()
        
        # Test 6: Statistics tracking
        success6 = test_statistics_tracking()
        
        # Test 7: Error handling
        success7 = test_error_handling()
        
        # Test 8: Configuration updates
        success8 = test_configuration_updates()
        
        if all([success1, success2, success3, success4, success5, success6, success7, success8]):
            logger.info("🎉 All ActionTranslator tests passed!")
            logger.info("✅ ActionTranslator system is working correctly")
            return True
        else:
            logger.error("❌ Some ActionTranslator tests failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)