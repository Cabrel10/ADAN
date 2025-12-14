#!/usr/bin/env python3
"""
Diagnostic script for data integrity analysis.

Performs root cause analysis on indicator calculations and data corruption issues.
Compares calculated indicators against Binance reference data and generates detailed reports.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

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


class DataIntegrityDiagnostic:
    """Diagnose data integrity issues in indicator calculations."""
    
    def __init__(self):
        self.calculator = IndicatorCalculator()
        self.validator = DataValidator()
        self.exchange = None
        self.results = {}
    
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
    
    def fetch_market_data(self, symbol='BTC/USDT', timeframe='5m', limit=100):
        """Fetch market data from Binance."""
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
    
    def analyze_indicators(self, df):
        """Analyze indicator calculations."""
        try:
            logger.info("🔍 Analyzing indicators...")
            
            indicators = self.calculator.calculate_all(df)
            
            analysis = {
                'rsi': {
                    'value': indicators['rsi'],
                    'range': [0, 100],
                    'status': 'valid' if 0 <= indicators['rsi'] <= 100 else 'invalid'
                },
                'adx': {
                    'value': indicators['adx'],
                    'range': [0, 100],
                    'status': 'valid' if 0 <= indicators['adx'] <= 100 else 'invalid'
                },
                'atr': {
                    'value': indicators['atr'],
                    'range': [0, float('inf')],
                    'status': 'valid' if indicators['atr'] > 0 else 'invalid'
                },
                'atr_percent': {
                    'value': indicators['atr_percent'],
                    'range': [0, 100],
                    'status': 'valid' if 0 <= indicators['atr_percent'] <= 100 else 'invalid'
                }
            }
            
            logger.info(f"✅ Indicators analyzed:")
            for name, data in analysis.items():
                logger.info(f"  {name}: {data['value']:.4f} ({data['status']})")
            
            return analysis, indicators
        except Exception as e:
            logger.error(f"❌ Indicator analysis failed: {e}")
            return None, None
    
    def validate_against_reference(self, indicators):
        """Validate indicators against Binance reference."""
        try:
            logger.info("🔐 Validating against Binance reference...")
            
            result = self.validator.validate_full_pipeline(
                calculated_indicators=indicators
            )
            
            validation_report = {
                'status': result.status,
                'message': result.message,
                'deviations': {
                    'rsi': result.rsi_deviation,
                    'adx': result.adx_deviation,
                    'atr': result.atr_deviation
                },
                'thresholds': {
                    'warning': self.validator.WARNING_THRESHOLD,
                    'halt': self.validator.HALT_THRESHOLD
                }
            }
            
            logger.info(f"✅ Validation result: {result.status}")
            logger.info(f"  RSI deviation: {result.rsi_deviation:.2f}%")
            logger.info(f"  ADX deviation: {result.adx_deviation:.2f}%")
            logger.info(f"  ATR deviation: {result.atr_deviation:.2f}%")
            
            return validation_report
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return None
    
    def check_data_quality(self, df):
        """Check data quality metrics."""
        try:
            logger.info("📈 Checking data quality...")
            
            quality = {
                'total_candles': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'price_range': {
                    'min': float(df['close'].min()),
                    'max': float(df['close'].max()),
                    'mean': float(df['close'].mean()),
                    'std': float(df['close'].std())
                },
                'volume_stats': {
                    'min': float(df['volume'].min()),
                    'max': float(df['volume'].max()),
                    'mean': float(df['volume'].mean())
                },
                'timestamp_gaps': self._check_timestamp_gaps(df)
            }
            
            logger.info(f"✅ Data quality checked:")
            logger.info(f"  Total candles: {quality['total_candles']}")
            logger.info(f"  Missing values: {quality['missing_values']}")
            logger.info(f"  Price range: {quality['price_range']['min']:.2f} - {quality['price_range']['max']:.2f}")
            
            return quality
        except Exception as e:
            logger.error(f"❌ Data quality check failed: {e}")
            return None
    
    def _check_timestamp_gaps(self, df):
        """Check for gaps in timestamp data."""
        gaps = []
        if len(df) > 1:
            time_diffs = df['timestamp'].diff()
            expected_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            
            for i, diff in enumerate(time_diffs):
                if diff != expected_diff and pd.notna(diff):
                    gaps.append({
                        'index': i,
                        'expected': str(expected_diff),
                        'actual': str(diff)
                    })
        
        return gaps
    
    def generate_report(self, symbol='BTC/USDT', timeframe='5m'):
        """Generate comprehensive diagnostic report."""
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INTEGRITY DIAGNOSTIC REPORT")
        logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"{'='*60}\n")
        
        # Fetch data
        df = self.fetch_market_data(symbol, timeframe)
        if df is None:
            return None
        
        # Check data quality
        quality = self.check_data_quality(df)
        
        # Analyze indicators
        analysis, indicators = self.analyze_indicators(df)
        if analysis is None:
            return None
        
        # Validate against reference
        validation = self.validate_against_reference(indicators)
        
        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'data_quality': quality,
            'indicator_analysis': analysis,
            'validation': validation,
            'summary': self._generate_summary(quality, analysis, validation)
        }
        
        return report
    
    def _generate_summary(self, quality, analysis, validation):
        """Generate summary of findings."""
        issues = []
        
        # Check data quality
        if quality and quality['missing_values']:
            for col, count in quality['missing_values'].items():
                if count > 0:
                    issues.append(f"Missing {count} values in {col}")
        
        # Check indicator validity
        if analysis:
            for name, data in analysis.items():
                if data['status'] != 'valid':
                    issues.append(f"Invalid {name}: {data['value']}")
        
        # Check validation
        if validation:
            if validation['status'] == 'halt':
                issues.append(f"HALT: {validation['message']}")
            elif validation['status'] == 'warning':
                issues.append(f"WARNING: {validation['message']}")
        
        return {
            'status': 'healthy' if not issues else 'issues_detected',
            'issues': issues,
            'issue_count': len(issues)
        }
    
    def export_report(self, report, filepath=None):
        """Export report to JSON file."""
        if filepath is None:
            filepath = f"data_integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"✅ Report exported to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Failed to export report: {e}")
            return None


def main():
    """Main execution."""
    diagnostic = DataIntegrityDiagnostic()
    
    if not diagnostic.setup_exchange():
        return
    
    # Generate report
    report = diagnostic.generate_report()
    
    if report:
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Status: {report['summary']['status']}")
        logger.info(f"Issues found: {report['summary']['issue_count']}")
        
        if report['summary']['issues']:
            logger.info("\nIssues:")
            for issue in report['summary']['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ No issues detected")
        
        # Export report
        diagnostic.export_report(report)


if __name__ == "__main__":
    main()
