#!/usr/bin/env python3
"""
Manual validation script for indicators against Binance.

Allows manual verification that calculated indicators match Binance reference values.
Useful for debugging and verification during development.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from adan_trading_bot.indicators.calculator import IndicatorCalculator
from adan_trading_bot.validation.data_validator import DataValidator
from adan_trading_bot.exchange_api.connector import get_exchange_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndicatorValidator:
    """Validate indicators against Binance reference data."""
    
    def __init__(self):
        self.calculator = IndicatorCalculator()
        self.validator = DataValidator()
        self.exchange = None
    
    def setup_exchange(self):
        """Initialize exchange connection."""
        try:
            config = {'paper_trading': {}}
            self.exchange = get_exchange_client(config)
            logger.info("✅ Exchange connected to Binance")
            return True
        except Exception as e:
            logger.error(f"❌ Exchange connection failed: {e}")
            return False
    
    def validate_symbol(self, symbol='BTC/USDT', timeframe='5m', limit=100):
        """Validate indicators for a specific symbol."""
        logger.info(f"\n{'='*70}")
        logger.info(f"INDICATOR VALIDATION: {symbol} {timeframe}")
        logger.info(f"{'='*70}\n")
        
        try:
            # Fetch data
            logger.info(f"📊 Fetching {limit} candles from Binance...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"✅ Fetched {len(df)} candles")
            logger.info(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            
            # Calculate indicators
            logger.info(f"\n🔧 Calculating indicators...")
            indicators = self.calculator.calculate_all(df)
            
            logger.info(f"✅ Indicators calculated:")
            logger.info(f"   RSI (14):     {indicators['rsi']:.2f}")
            logger.info(f"   ADX (14):     {indicators['adx']:.2f}")
            logger.info(f"   ATR (14):     {indicators['atr']:.6f}")
            logger.info(f"   ATR %:        {indicators['atr_percent']:.4f}%")
            
            # Validate against reference
            logger.info(f"\n🔐 Validating against Binance reference...")
            result = self.validator.validate_full_pipeline(
                calculated_indicators=indicators
            )
            
            logger.info(f"✅ Validation result: {result.status.upper()}")
            logger.info(f"   Message: {result.message}")
            logger.info(f"\n   Deviations:")
            logger.info(f"   RSI:  {result.rsi_deviation:+.2f}%")
            logger.info(f"   ADX:  {result.adx_deviation:+.2f}%")
            logger.info(f"   ATR:  {result.atr_deviation:+.2f}%")
            
            # Thresholds
            logger.info(f"\n   Thresholds:")
            logger.info(f"   Warning: {self.validator.WARNING_THRESHOLD}%")
            logger.info(f"   Halt:    {self.validator.HALT_THRESHOLD}%")
            
            # Status
            if result.status == "halt":
                logger.critical(f"\n🚨 HALT: Indicators deviate too much from reference!")
                return False
            elif result.status == "warning":
                logger.warning(f"\n⚠️  WARNING: Indicators show significant deviation")
                return True
            else:
                logger.info(f"\n✅ PASS: Indicators match reference data")
                return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def validate_multiple_symbols(self, symbols=None, timeframe='5m'):
        """Validate multiple symbols."""
        if symbols is None:
            symbols = ['BTC/USDT', 'ETH/USDT']
        
        results = {}
        for symbol in symbols:
            results[symbol] = self.validate_symbol(symbol, timeframe)
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'='*70}")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        logger.info(f"Passed: {passed}/{total}")
        for symbol, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {symbol}: {status}")
        
        return all(results.values())


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Validate indicators against Binance')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe')
    parser.add_argument('--limit', type=int, default=100, help='Number of candles')
    parser.add_argument('--symbols', type=str, nargs='+', help='Multiple symbols to validate')
    
    args = parser.parse_args()
    
    validator = IndicatorValidator()
    
    if not validator.setup_exchange():
        return
    
    if args.symbols:
        validator.validate_multiple_symbols(args.symbols, args.timeframe)
    else:
        validator.validate_symbol(args.symbol, args.timeframe, args.limit)


if __name__ == "__main__":
    main()
