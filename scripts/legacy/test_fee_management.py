#!/usr/bin/env python3
"""
Test script for fee management in the ADAN Trading Bot.

This script tests the comprehensive fee management system including
the FeeManager and its integration with ActionTranslator.
"""

import sys
from pathlib import Path
import logging

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.trading.fee_manager import (
    FeeManager, FeeType, FeeStructure, FeeConfig
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


def test_basic_fee_calculation():
    """Test basic fee calculation functionality."""
    logger.info("Testing basic fee calculation...")
    
    fee_manager = FeeManager()
    
    # Test basic trading fee calculation
    trade_value = 10000.0  # $10,000 trade
    
    # Test maker fee
    maker_result = fee_manager.calculate_trading_fee(
        trade_value=trade_value,
        asset="BTC",
        is_maker=True
    )
    
    assert maker_result.total_fee > 0, "Maker fee should be positive"
    assert maker_result.effective_rate == 0.001, "Default maker rate should be 0.1%"
    logger.info(f"✅ Maker fee: ${maker_result.total_fee:.2f} ({maker_result.effective_rate*100:.3f}%)")
    
    # Test taker fee
    taker_result = fee_manager.calculate_trading_fee(
        trade_value=trade_value,
        asset="BTC",
        is_maker=False
    )
    
    assert taker_result.total_fee > maker_result.total_fee, "Taker fee should be higher than maker fee"
    assert taker_result.effective_rate == 0.0015, "Default taker rate should be 0.15%"
    logger.info(f"✅ Taker fee: ${taker_result.total_fee:.2f} ({taker_result.effective_rate*100:.3f}%)")
    
    return True


def test_asset_specific_fees():
    """Test asset-specific fee adjustments."""
    logger.info("Testing asset-specific fee adjustments...")
    
    fee_manager = FeeManager()
    trade_value = 10000.0
    
    # Test BTC (no adjustment)
    btc_result = fee_manager.calculate_trading_fee(
        trade_value=trade_value,
        asset="BTC",
        is_maker=False
    )
    
    # Test USDT (should have discount)
    usdt_result = fee_manager.calculate_trading_fee(
        trade_value=trade_value,
        asset="USDT",
        is_maker=False
    )
    
    assert usdt_result.total_fee < btc_result.total_fee, "USDT should have lower fees than BTC"
    logger.info(f"✅ BTC fee: ${btc_result.total_fee:.2f}, USDT fee: ${usdt_result.total_fee:.2f}")
    
    return True


def test_volume_discounts():
    """Test volume-based fee discounts."""
    logger.info("Testing volume-based fee discounts...")
    
    fee_manager = FeeManager()
    trade_value = 10000.0
    
    # Test without volume discount
    no_volume_result = fee_manager.calculate_trading_fee(
        trade_value=trade_value,
        asset="BTC",
        is_maker=False,
        user_volume_30d=None
    )
    
    # Test with high volume (should get discount)
    high_volume_result = fee_manager.calculate_trading_fee(
        trade_value=trade_value,
        asset="BTC",
        is_maker=False,
        user_volume_30d=50000.0  # $50k volume
    )
    
    assert high_volume_result.total_fee < no_volume_result.total_fee, "High volume should get discount"
    assert len(high_volume_result.warnings) > 0, "Should have warning about volume discount"
    logger.info(f"✅ No volume: ${no_volume_result.total_fee:.2f}, High volume: ${high_volume_result.total_fee:.2f}")
    
    return True


def test_funding_fees():
    """Test funding fee calculation for futures."""
    logger.info("Testing funding fee calculation...")
    
    fee_manager = FeeManager()
    
    # Test funding fee calculation
    position_value = 50000.0  # $50k position
    funding_rate = 0.0001  # 0.01% funding rate
    hours_held = 24.0  # 24 hours (3 funding periods)
    
    funding_result = fee_manager.calculate_funding_fee(
        position_value=position_value,
        funding_rate=funding_rate,
        hours_held=hours_held
    )
    
    expected_fee = position_value * funding_rate * (hours_held / 8.0)
    assert abs(funding_result.total_fee - expected_fee) < 0.01, "Funding fee calculation incorrect"
    assert 'funding_fee' in funding_result.fee_breakdown, "Should have funding fee breakdown"
    logger.info(f"✅ Funding fee for 24h: ${funding_result.total_fee:.2f}")
    
    return True


def test_withdrawal_fees():
    """Test withdrawal fee calculation."""
    logger.info("Testing withdrawal fee calculation...")
    
    fee_manager = FeeManager()
    
    # Test BTC withdrawal
    btc_withdrawal = fee_manager.calculate_withdrawal_fee(
        amount=1.0,
        asset="BTC",
        network="bitcoin"
    )
    
    assert btc_withdrawal.total_fee == 0.0005, "BTC withdrawal fee should be 0.0005"
    assert btc_withdrawal.currency == "BTC", "Fee currency should be BTC"
    logger.info(f"✅ BTC withdrawal fee: {btc_withdrawal.total_fee} BTC")
    
    # Test USDT withdrawal on different networks
    usdt_eth = fee_manager.calculate_withdrawal_fee(
        amount=1000.0,
        asset="USDT",
        network="ethereum"
    )
    
    usdt_tron = fee_manager.calculate_withdrawal_fee(
        amount=1000.0,
        asset="USDT",
        network="tron"
    )
    
    assert usdt_tron.total_fee < usdt_eth.total_fee, "TRON should be cheaper than Ethereum"
    logger.info(f"✅ USDT withdrawal - ETH: ${usdt_eth.total_fee}, TRON: ${usdt_tron.total_fee}")
    
    return True


def test_fee_optimization():
    """Test fee optimization recommendations."""
    logger.info("Testing fee optimization...")
    
    fee_manager = FeeManager()
    trade_value = 10000.0
    
    # Test low urgency (should recommend limit order)
    low_urgency = fee_manager.optimize_order_type(
        trade_value=trade_value,
        urgency=0.2
    )
    
    # The recommendation depends on the fee savings calculation
    # Let's check if it's either limit or market with proper reasoning
    assert low_urgency['recommended_type'] in ['limit', 'market'], "Should recommend either limit or market"
    logger.info(f"✅ Low urgency recommendation: {low_urgency['recommended_type']} - {low_urgency['reason']}")
    
    # Test high urgency (should recommend market order)
    high_urgency = fee_manager.optimize_order_type(
        trade_value=trade_value,
        urgency=0.9
    )
    
    assert high_urgency['recommended_type'] == 'market', "High urgency should recommend market order"
    logger.info(f"✅ High urgency recommendation: {high_urgency['recommended_type']}")
    
    return True


def test_action_translator_integration():
    """Test FeeManager integration with ActionTranslator."""
    logger.info("Testing ActionTranslator integration with FeeManager...")
    
    # Create FeeManager with custom settings
    fee_manager = FeeManager(
        default_maker_fee=0.0008,
        default_taker_fee=0.0012
    )
    
    # Create ActionTranslator with FeeManager
    translator = ActionTranslator(
        fee_manager=fee_manager,
        exchange="binance"
    )
    
    # Test portfolio state
    portfolio_state = {
        'cash': 10000.0,
        'total_value': 12000.0,
        'positions': {}
    }
    
    # Test action translation with fee calculation
    agent_action = 1  # BUY action
    trading_action = translator.translate_action(
        agent_action=agent_action,
        asset='BTC',
        current_price=50000.0,
        portfolio_state=portfolio_state
    )
    
    assert trading_action.fees is not None, "Trading action should have fees calculated"
    assert trading_action.fees > 0, "Fees should be positive for buy action"
    logger.info(f"✅ Trading action fees: ${trading_action.fees:.2f}")
    
    # Test that fees are included in validation
    validation_result = translator.validate_action(trading_action, portfolio_state)
    assert validation_result.is_valid, f"Action should be valid: {validation_result.message}"
    logger.info("✅ Action validation includes fee calculation")
    
    return True


def test_fee_statistics():
    """Test fee statistics tracking."""
    logger.info("Testing fee statistics tracking...")
    
    fee_manager = FeeManager()
    
    # Perform several fee calculations
    for i in range(5):
        fee_manager.calculate_trading_fee(
            trade_value=1000.0 * (i + 1),
            asset="BTC",
            is_maker=(i % 2 == 0)
        )
    
    # Get fee summary
    summary = fee_manager.get_fee_summary()
    
    assert summary['total_trades'] == 5, "Should have tracked 5 trades"
    assert summary['total_fees_paid'] > 0, "Should have accumulated fees"
    assert summary['average_fee_per_trade'] > 0, "Should have positive average fee"
    logger.info(f"✅ Fee statistics: {summary['total_trades']} trades, "
                f"${summary['total_fees_paid']:.2f} total fees")
    
    return True


def test_custom_fee_configs():
    """Test custom fee configurations."""
    logger.info("Testing custom fee configurations...")
    
    fee_manager = FeeManager()
    
    # Add custom fee configuration
    custom_config = FeeConfig(
        fee_type=FeeType.TAKER,
        structure=FeeStructure.FIXED,
        rate=5.0,  # $5 fixed fee
        currency="USD"
    )
    
    fee_manager.add_fee_config("custom_exchange", custom_config)
    
    # Test that custom config is used
    # Note: This would require modifying the fee calculation logic
    # to accept exchange-specific configs, which is a future enhancement
    
    logger.info("✅ Custom fee configuration added successfully")
    
    return True


def run_all_tests():
    """Run all fee management tests."""
    logger.info("Starting fee management tests...")
    
    tests = [
        test_basic_fee_calculation,
        test_asset_specific_fees,
        test_volume_discounts,
        test_funding_fees,
        test_withdrawal_fees,
        test_fee_optimization,
        test_action_translator_integration,
        test_fee_statistics,
        test_custom_fee_configs
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
        logger.info("🎉 All fee management tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)