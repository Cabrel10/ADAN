#!/usr/bin/env python3
"""
Test indicator reproducibility and consistency.

Verifies that indicator calculations are deterministic and reproducible
across multiple runs with the same input data.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from adan_trading_bot.indicators.calculator import IndicatorCalculator
from adan_trading_bot.exchange_api.connector import get_exchange_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReproducibilityTester:
    """Test indicator calculation reproducibility."""
    
    def __init__(self):
        self.calculator = IndicatorCalculator()
        self.exchange = None
    
    def setup_exchange(self):
        """Initialize exchange connection."""
        try:
            config = {'paper_trading': {}}
            self.exchange = get_exchange_client(config)
            logger.info("✅ Exchange connected")
            return True
        except Exception as e:
            logger.error(f"❌ Exchange connection failed: {e}")
            return False
    
    def fetch_data(self, symbol='BTC/USDT', timeframe='5m', limit=100):
        """Fetch market data."""
        try:
            logger.info(f"📊 Fetching {symbol} {timeframe} data...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"✅ Fetched {len(df)} candles")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to fetch data: {e}")
            return None
    
    def test_determinism(self, df, num_runs=5):
        """Test that calculations are deterministic."""
        logger.info(f"\n{'='*70}")
        logger.info(f"DETERMINISM TEST ({num_runs} runs)")
        logger.info(f"{'='*70}\n")
        
        results = []
        
        for i in range(num_runs):
            logger.info(f"Run {i+1}/{num_runs}...")
            indicators = self.calculator.calculate_all(df)
            results.append(indicators)
            logger.info(f"  RSI: {indicators['rsi']:.6f}")
            logger.info(f"  ADX: {indicators['adx']:.6f}")
            logger.info(f"  ATR: {indicators['atr']:.10f}")
        
        # Check consistency
        logger.info(f"\n🔍 Checking consistency...")
        
        all_consistent = True
        for key in ['rsi', 'adx', 'atr', 'atr_percent']:
            values = [r[key] for r in results]
            
            # Check if all values are identical
            if len(set(values)) == 1:
                logger.info(f"✅ {key}: Consistent across all runs")
            else:
                # Check if values are very close (floating point tolerance)
                max_diff = max(values) - min(values)
                if max_diff < 1e-10:
                    logger.info(f"✅ {key}: Consistent (max diff: {max_diff:.2e})")
                else:
                    logger.warning(f"⚠️  {key}: Inconsistent (max diff: {max_diff:.2e})")
                    logger.warning(f"   Values: {values}")
                    all_consistent = False
        
        return all_consistent
    
    def test_stability(self, symbol='BTC/USDT', timeframe='5m', num_fetches=3):
        """Test that calculations are stable across multiple data fetches."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STABILITY TEST ({num_fetches} fetches)")
        logger.info(f"{'='*70}\n")
        
        results = []
        
        for i in range(num_fetches):
            logger.info(f"Fetch {i+1}/{num_fetches}...")
            df = self.fetch_data(symbol, timeframe)
            if df is None:
                return False
            
            indicators = self.calculator.calculate_all(df)
            results.append(indicators)
            logger.info(f"  RSI: {indicators['rsi']:.6f}")
            logger.info(f"  ADX: {indicators['adx']:.6f}")
            logger.info(f"  ATR: {indicators['atr']:.10f}")
        
        # Check stability
        logger.info(f"\n🔍 Checking stability...")
        
        all_stable = True
        for key in ['rsi', 'adx', 'atr', 'atr_percent']:
            values = [r[key] for r in results]
            
            # Calculate variance
            variance = np.var(values)
            mean = np.mean(values)
            
            if variance < 1e-10:
                logger.info(f"✅ {key}: Stable (variance: {variance:.2e})")
            else:
                # Check if variance is reasonable (< 1% of mean)
                if mean > 0:
                    cv = np.sqrt(variance) / mean
                    if cv < 0.01:
                        logger.info(f"✅ {key}: Stable (CV: {cv:.4f})")
                    else:
                        logger.warning(f"⚠️  {key}: Unstable (CV: {cv:.4f})")
                        all_stable = False
                else:
                    logger.warning(f"⚠️  {key}: Cannot calculate CV (mean={mean})")
        
        return all_stable
    
    def test_edge_cases(self, df):
        """Test edge cases."""
        logger.info(f"\n{'='*70}")
        logger.info("EDGE CASE TESTS")
        logger.info(f"{'='*70}\n")
        
        all_passed = True
        
        # Test 1: Minimum data (30 candles)
        logger.info("Test 1: Minimum data (30 candles)...")
        try:
            min_df = df.iloc[-30:].copy()
            indicators = self.calculator.calculate_all(min_df)
            logger.info(f"✅ Passed: RSI={indicators['rsi']:.2f}, ADX={indicators['adx']:.2f}")
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            all_passed = False
        
        # Test 2: Flat market (all same price)
        logger.info("\nTest 2: Flat market (all same price)...")
        try:
            flat_df = df.copy()
            flat_df['open'] = 50000
            flat_df['high'] = 50000
            flat_df['low'] = 50000
            flat_df['close'] = 50000
            
            indicators = self.calculator.calculate_all(flat_df)
            logger.info(f"✅ Passed: RSI={indicators['rsi']:.2f}, ADX={indicators['adx']:.2f}, ATR={indicators['atr']:.6f}")
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            all_passed = False
        
        # Test 3: Extreme volatility
        logger.info("\nTest 3: Extreme volatility...")
        try:
            volatile_df = df.copy()
            volatile_df['high'] = volatile_df['close'] * 1.5
            volatile_df['low'] = volatile_df['close'] * 0.5
            
            indicators = self.calculator.calculate_all(volatile_df)
            logger.info(f"✅ Passed: RSI={indicators['rsi']:.2f}, ADX={indicators['adx']:.2f}, ATR={indicators['atr']:.6f}")
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            all_passed = False
        
        return all_passed
    
    def run_all_tests(self, symbol='BTC/USDT', timeframe='5m'):
        """Run all reproducibility tests."""
        logger.info(f"\n{'='*70}")
        logger.info("INDICATOR REPRODUCIBILITY TEST SUITE")
        logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"{'='*70}\n")
        
        # Fetch data
        df = self.fetch_data(symbol, timeframe)
        if df is None:
            return False
        
        # Run tests
        test_results = {}
        
        test_results['determinism'] = self.test_determinism(df)
        test_results['stability'] = self.test_stability(symbol, timeframe)
        test_results['edge_cases'] = self.test_edge_cases(df)
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*70}")
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        all_passed = all(test_results.values())
        
        if all_passed:
            logger.info(f"\n✅ All tests passed!")
        else:
            logger.warning(f"\n⚠️  Some tests failed")
        
        return all_passed


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Test indicator reproducibility')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe')
    
    args = parser.parse_args()
    
    tester = ReproducibilityTester()
    
    if not tester.setup_exchange():
        return
    
    tester.run_all_tests(args.symbol, args.timeframe)


if __name__ == "__main__":
    main()
