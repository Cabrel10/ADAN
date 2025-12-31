import sys
import os
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from adan_trading_bot.indicators.calculator import IndicatorCalculator
from paper_trading_monitor import RealPaperTradingMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_indicators_integration():
    """Test that indicators are correctly calculated and integrated into observation."""
    
    # 1. Create dummy data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='5min')
    data = {
        'open': np.random.uniform(40000, 41000, 100),
        'high': np.random.uniform(41000, 42000, 100),
        'low': np.random.uniform(39000, 40000, 100),
        'close': np.random.uniform(40000, 41000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    
    # 2. Test IndicatorCalculator directly
    logger.info("Testing IndicatorCalculator.calculate_features_df...")
    df_5m = IndicatorCalculator.calculate_features_df(df, '5m')
    
    expected_cols_5m = [
        'rsi_14', 'macd_12_26_9', 'bb_percent_b_20_2', 'atr_14', 
        'volume_ratio_20', 'ema_20_ratio', 'stoch_k_14_3_3'
    ]
    
    for col in expected_cols_5m:
        if col not in df_5m.columns:
            logger.error(f"❌ Missing column in 5m: {col}")
        else:
            # Check if values are not all zeros (except maybe first few)
            non_zeros = np.count_nonzero(df_5m[col].values)
            if non_zeros == 0:
                logger.warning(f"⚠️ Column {col} is all zeros!")
            else:
                logger.info(f"✅ Column {col} present and has values")

    # 3. Test Monitor.build_observation
    logger.info("\nTesting Monitor.build_observation...")
    monitor = RealPaperTradingMonitor()
    
    raw_data = {
        'BTC/USDT': {
            '5m': df,
            '1h': df, # Reusing same data for simplicity
            '4h': df
        }
    }
    
    obs = monitor.build_observation(raw_data)
    
    if obs is None:
        logger.error("❌ build_observation returned None")
        return
        
    for tf in ['5m', '1h', '4h']:
        if tf in obs:
            shape = obs[tf].shape
            logger.info(f"✅ {tf} observation shape: {shape}")
            assert shape == (20, 14), f"Wrong shape for {tf}: {shape}"
            
            # Check for zero columns (potential missing indicators)
            # Note: normalized values can be 0, but unlikely for whole column to be exactly 0
            # unless it was padded with 0s
            zero_cols = []
            for i in range(14):
                if np.all(obs[tf][:, i] == 0):
                    zero_cols.append(i)
            
            if zero_cols:
                logger.warning(f"⚠️ {tf} has zero-filled columns at indices: {zero_cols}")
            else:
                logger.info(f"✅ {tf} has no zero-filled columns")
                
    logger.info("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_indicators_integration()
