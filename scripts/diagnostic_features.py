#!/usr/bin/env python3
"""
🔬 DIAGNOSTIC: Check for NaN/corruption in live features
Analyzes feature distributions to detect data quality issues
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer
from adan_trading_bot.data_processing.data_fetcher import DataFetcher
from config.config_loader import load_config

def check_live_features():
    """Fetch live data and check for feature corruption"""
    
    print("\n🔬 LIVE FEATURE ANALYSIS")
    print("=" * 80)
    
    try:
        # Load config
        config = load_config("config/config.yaml")
        
        # Fetch live data
        print("\n1️⃣ Fetching live data...")
        fetcher = DataFetcher(exchange="binance", testnet=True)
        
        timeframes = ['5m', '1h', '4h']
        data = {}
        
        for tf in timeframes:
            df = fetcher.fetch_ohlcv_data('BTC/USDT', timeframe=tf, limit=500)
            print(f"   ✅ {tf}: {len(df)} candles")
            
            # Check for NaN in raw OHLCV
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                print(f"   ⚠️ {nan_count} NaN values in raw OHLCV data!")
            
            data[tf] = df
        
        # Calculate indicators
        print("\n2️⃣ Calculating indicators...")
        engineer = FeatureEngineer(config)
        
        for tf in timeframes:
            print(f"\n   Analyzing {tf}:")
            df_ind = engineer.calculate_indicators_for_single_timeframe(data[tf], tf)
            
            # Statistics per column
            for col in df_ind.columns:
                values = df_ind[col].dropna()
                nan_pct = (df_ind[col].isna().sum() / len(df_ind)) * 100
                
                if len(values) == 0:
                    print(f"      ❌ {col}: ALL NaN")
                    continue
                
                # Check for suspicious patterns
                issues = []
                
                if values.min() == 0 and values.max() == 0:
                    issues.append("ALL ZEROS")
                
                if values.std() < 0.001:
                    issues.append("NO VARIANCE")
                
                if nan_pct > 50:
                    issues.append(f"{nan_pct:.0f}% NaN")
                
                if issues:
                    print(f"      ⚠️ {col}: {', '.join(issues)}")
                    print(f"         → Min={values.min():.4f}, Max={values.max():.4f}, "
                          f"Mean={values.mean():.4f}, Std={values.std():.4f}")
        
        print("\n" + "=" * 80)
        print("✅ Analysis complete. Check warnings above for data corruption.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_live_features()
