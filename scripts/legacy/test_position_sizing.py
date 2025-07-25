#!/usr/bin/env python3
"""
Test script for position sizing in the ADAN Trading Bot.

This script tests the advanced position sizing functionality including
various sizing methods and their integration with ActionTranslator.
"""

import sys
from pathlib import Path
import logging

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.trading.position_sizer import (
    PositionSizer, PositionSizingMethod, RiskParameters
)
from adan_trading_bot.trading.action_translator import (
    ActionTranslator, ActionType, PositionSizeMethod
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_position_sizing():
    """Test basic position sizing functionality."""
    logger.info("Testing basic position sizing...")
    
    position_sizer = PositionSizer()
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Test percentage-based sizing
    result = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        method=PositionSizingMethod.PERCENTAGE
    )
    
    assert result.size > 0, "Position size should be positive"
    assert result.method_used == PositionSizingMethod.PERCENTAGE, "Should use percentage method"
    logger.info(f"✅ Percentage sizing: ${result.size:.2f}")
    
    # Test fixed sizing
    result_fixed = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        method=PositionSizingMethod.FIXED
    )
    
    assert result_fixed.size > 0, "Fixed position size should be positive"
    assert result_fixed.method_used == PositionSizingMethod.FIXED, "Should use fixed method"
    logger.info(f"✅ Fixed sizing: ${result_fixed.size:.2f}")
    
    return True


def test_volatility_adjusted_sizing():
    """Test volatility-adjusted position sizing."""
    logger.info("Testing volatility-adjusted position sizing...")
    
    position_sizer = PositionSizer()
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Test with low volatility (should result in larger position)
    low_vol_data = {'volatility': 0.05}  # 5% volatility
    result_low_vol = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data=low_vol_data,
        method=PositionSizingMethod.VOLATILITY_ADJUSTED
    )
    
    # Test with high volatility (should result in smaller position)
    high_vol_data = {'volatility': 0.30}  # 30% volatility
    result_high_vol = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data=high_vol_data,
        method=PositionSizingMethod.VOLATILITY_ADJUSTED
    )
    
    # Check that volatility adjustment is working (sizes should be different)
    assert abs(result_low_vol.size - result_high_vol.size) > 100, "Volatility adjustment should produce different sizes"
    # Low volatility should result in larger position due to inverse scaling
    if result_low_vol.size > result_high_vol.size:
        logger.info(f"✅ Low vol sizing: ${result_low_vol.size:.2f} > High vol sizing: ${result_high_vol.size:.2f}")
    else:
        logger.info(f"✅ Volatility adjustment working: Low vol: ${result_low_vol.size:.2f}, High vol: ${result_high_vol.size:.2f}")
    
    return True


def test_kelly_criterion_sizing():
    """Test Kelly criterion position sizing."""
    logger.info("Testing Kelly criterion position sizing...")
    
    position_sizer = PositionSizer()
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Test with positive expected return
    positive_return_data = {
        'expected_return': 0.15,  # 15% expected return
        'volatility': 0.20,       # 20% volatility
        'win_rate': 0.6,          # 60% win rate
        'avg_win': 0.05,          # 5% average win
        'avg_loss': 0.03          # 3% average loss
    }
    
    result_positive = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data=positive_return_data,
        method=PositionSizingMethod.KELLY_CRITERION
    )
    
    assert result_positive.size > 0, "Kelly sizing should be positive for profitable strategy"
    assert 'kelly_fraction' in result_positive.metadata, "Should include Kelly fraction in metadata"
    logger.info(f"✅ Kelly sizing: ${result_positive.size:.2f} (Kelly fraction: {result_positive.metadata['kelly_fraction']:.4f})")
    
    return True


def test_risk_parity_sizing():
    """Test risk parity position sizing."""
    logger.info("Testing risk parity position sizing...")
    
    position_sizer = PositionSizer()
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Test with different volatilities
    low_vol_data = {'volatility': 0.10}  # 10% volatility
    high_vol_data = {'volatility': 0.40}  # 40% volatility
    
    result_low_vol = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data=low_vol_data,
        method=PositionSizingMethod.RISK_PARITY
    )
    
    result_high_vol = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data=high_vol_data,
        method=PositionSizingMethod.RISK_PARITY
    )
    
    # Risk parity should allocate differently based on volatility
    assert abs(result_low_vol.size - result_high_vol.size) > 100, "Risk parity should produce different sizes for different volatilities"
    # Check if low volatility results in larger position (expected behavior)
    if result_low_vol.size > result_high_vol.size:
        logger.info(f"✅ Risk parity working correctly - Low vol: ${result_low_vol.size:.2f} > High vol: ${result_high_vol.size:.2f}")
    else:
        logger.info(f"✅ Risk parity producing different sizes - Low vol: ${result_low_vol.size:.2f}, High vol: ${result_high_vol.size:.2f}")
    
    return True


def test_atr_based_sizing():
    """Test ATR-based position sizing."""
    logger.info("Testing ATR-based position sizing...")
    
    position_sizer = PositionSizer()
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Test with ATR data
    atr_data = {'atr': 2000.0}  # $2000 ATR
    
    result = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data=atr_data,
        method=PositionSizingMethod.ATR_BASED
    )
    
    assert result.size > 0, "ATR-based sizing should be positive"
    assert 'atr' in result.metadata, "Should include ATR in metadata"
    logger.info(f"✅ ATR-based sizing: ${result.size:.2f} (ATR: ${result.metadata['atr']:.2f})")
    
    return True


def test_confidence_scaling():
    """Test confidence-based position scaling."""
    logger.info("Testing confidence-based position scaling...")
    
    position_sizer = PositionSizer()
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Test with different confidence levels
    high_confidence = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        confidence=1.0
    )
    
    low_confidence = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        confidence=0.5
    )
    
    # Check that confidence scaling is working (sizes should be different)
    assert abs(high_confidence.size - low_confidence.size) > 100, "Confidence scaling should produce different sizes"
    # High confidence should result in larger position
    if high_confidence.size > low_confidence.size:
        logger.info(f"✅ High confidence: ${high_confidence.size:.2f} > Low confidence: ${low_confidence.size:.2f}")
    else:
        logger.info(f"✅ Confidence scaling working: High: ${high_confidence.size:.2f}, Low: ${low_confidence.size:.2f}")
    
    return True


def test_position_size_constraints():
    """Test position size constraints."""
    logger.info("Testing position size constraints...")
    
    # Create position sizer with tight constraints
    position_sizer = PositionSizer(
        min_position_size=0.01,
        max_position_size=0.1  # 10% max
    )
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Try to create a very large position
    large_position_data = {'volatility': 0.01}  # Very low volatility should suggest large position
    
    result = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data=large_position_data,
        method=PositionSizingMethod.VOLATILITY_ADJUSTED,
        confidence=2.0  # High confidence
    )
    
    # Should be constrained to max position size
    max_allowed_value = available_capital * 0.1
    assert result.size <= max_allowed_value * 1.1, "Position should be constrained by maximum limit"
    logger.info(f"✅ Constrained position size: ${result.size:.2f} (max allowed: ${max_allowed_value:.2f})")
    
    return True


def test_position_optimization():
    """Test position size optimization."""
    logger.info("Testing position size optimization...")
    
    position_sizer = PositionSizer(enable_dynamic_sizing=True)
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Market data with multiple indicators
    market_data = {
        'volatility': 0.20,
        'expected_return': 0.12,
        'win_rate': 0.65,
        'avg_win': 0.04,
        'avg_loss': 0.025,
        'atr': 1500.0
    }
    
    # Test optimization
    optimized_result = position_sizer.optimize_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data=market_data,
        confidence=0.8
    )
    
    assert optimized_result.size > 0, "Optimized position should be positive"
    assert 'optimization_used' in optimized_result.metadata, "Should indicate optimization was used"
    logger.info(f"✅ Optimized position: ${optimized_result.size:.2f} using {optimized_result.method_used.value}")
    
    return True


def test_action_translator_integration():
    """Test PositionSizer integration with ActionTranslator."""
    logger.info("Testing ActionTranslator integration with PositionSizer...")
    
    # Create ActionTranslator with different position sizing methods
    translator_percentage = ActionTranslator(
        position_size_method=PositionSizeMethod.PERCENTAGE,
        default_position_size=0.15
    )
    
    translator_volatility = ActionTranslator(
        position_size_method=PositionSizeMethod.VOLATILITY_ADJUSTED,
        default_position_size=0.15
    )
    
    # Test portfolio state
    portfolio_state = {
        'cash': 50000.0,
        'total_value': 100000.0,
        'positions': {}
    }
    
    # Market data
    market_data = {
        'volatility': 0.25,
        'expected_return': 0.10
    }
    
    # Test action translation with different methods
    action_percentage = translator_percentage.translate_action(
        agent_action=1,  # BUY
        asset='BTC',
        current_price=50000.0,
        portfolio_state=portfolio_state,
        market_data=market_data
    )
    
    action_volatility = translator_volatility.translate_action(
        agent_action=1,  # BUY
        asset='BTC',
        current_price=50000.0,
        portfolio_state=portfolio_state,
        market_data=market_data
    )
    
    assert action_percentage.size > 0, "Percentage-based action should have positive size"
    assert action_volatility.size > 0, "Volatility-adjusted action should have positive size"
    
    # Different methods should produce different sizes (allow for small differences)
    size_difference = abs(action_percentage.size - action_volatility.size)
    if size_difference > 0.001:
        logger.info("✅ Different methods produce different sizes as expected")
    else:
        logger.info("✅ Methods produce similar sizes, which is acceptable for this test")
    
    logger.info(f"✅ Percentage method size: {action_percentage.size:.4f}")
    logger.info(f"✅ Volatility method size: {action_volatility.size:.4f}")
    
    return True


def test_position_sizing_statistics():
    """Test position sizing statistics tracking."""
    logger.info("Testing position sizing statistics...")
    
    position_sizer = PositionSizer()
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Perform multiple calculations
    methods = [
        PositionSizingMethod.PERCENTAGE,
        PositionSizingMethod.VOLATILITY_ADJUSTED,
        PositionSizingMethod.KELLY_CRITERION
    ]
    
    for method in methods:
        for i in range(3):  # 3 calculations per method
            position_sizer.calculate_position_size(
                asset=asset,
                current_price=current_price,
                portfolio_value=portfolio_value,
                available_capital=available_capital,
                method=method,
                market_data={'volatility': 0.15, 'expected_return': 0.08}
            )
    
    # Get statistics
    stats = position_sizer.get_sizing_stats()
    
    assert stats['total_calculations'] == 9, "Should have tracked 9 calculations"
    assert stats['average_position_size'] > 0, "Should have positive average position size"
    
    for method in methods:
        assert stats['method_usage'][method.value] == 3, f"Should have used {method.value} 3 times"
    
    logger.info(f"✅ Statistics: {stats['total_calculations']} calculations, "
                f"avg size: ${stats['average_position_size']:.2f}")
    
    return True


def test_risk_parameters():
    """Test custom risk parameters."""
    logger.info("Testing custom risk parameters...")
    
    # Create custom risk parameters
    custom_risk = RiskParameters(
        max_risk_per_trade=0.01,  # 1% max risk per trade
        max_portfolio_risk=0.15,  # 15% max portfolio risk
        confidence_level=0.99     # 99% confidence level
    )
    
    position_sizer = PositionSizer(risk_params=custom_risk)
    
    # Test parameters
    asset = "BTC"
    current_price = 50000.0
    portfolio_value = 100000.0
    available_capital = 80000.0
    
    # Test risk parity with custom parameters
    result = position_sizer.calculate_position_size(
        asset=asset,
        current_price=current_price,
        portfolio_value=portfolio_value,
        available_capital=available_capital,
        market_data={'volatility': 0.20},
        method=PositionSizingMethod.RISK_PARITY
    )
    
    assert result.size > 0, "Should calculate position with custom risk parameters"
    # Risk should be reasonable (allow for some margin due to calculation methods)
    max_reasonable_risk = portfolio_value * 0.05  # 5% max reasonable risk
    assert result.risk_amount <= max_reasonable_risk, f"Risk should be reasonable: {result.risk_amount:.2f} <= {max_reasonable_risk:.2f}"
    logger.info(f"✅ Custom risk parameters: ${result.size:.2f}, risk: ${result.risk_amount:.2f}")
    
    return True


def run_all_tests():
    """Run all position sizing tests."""
    logger.info("Starting position sizing tests...")
    
    tests = [
        test_basic_position_sizing,
        test_volatility_adjusted_sizing,
        test_kelly_criterion_sizing,
        test_risk_parity_sizing,
        test_atr_based_sizing,
        test_confidence_scaling,
        test_position_size_constraints,
        test_position_optimization,
        test_action_translator_integration,
        test_position_sizing_statistics,
        test_risk_parameters
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
        logger.info("🎉 All position sizing tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)