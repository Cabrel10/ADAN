#!/usr/bin/env python3
"""
Generate training scalers from training data parquets.
These scalers will be used in production to avoid distribution shift.
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.data_processing.state_builder import StateBuilder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def generate_scalers():
    """Generate and save training scalers."""
    
    # Paths
    data_dir = Path('/home/morningstar/Documents/trading/bot/data/processed/indicators/train/BTCUSDT')
    output_dir = Path('/home/morningstar/Documents/trading/bot/models')
    output_path = output_dir / 'training_scalers.pkl'
    
    # Create output dir if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🔧 GENERATING TRAINING SCALERS")
    print("=" * 80)
    
    scalers = {}
    
    # Load training data for each timeframe
    timeframes = ['5m', '1h', '4h']
    scaler_types = {
        '5m': MinMaxScaler(feature_range=(0, 1)),
        '1h': StandardScaler(),
        '4h': RobustScaler()
    }
    
    # Define training columns (same as in feature_engineer.py)
    train_columns = {
        '5m': ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 
               'macd_12_26_9',
               'bb_percent_b_20_2', 'atr_14', 'volume_ratio_20', 
               'ema_20_ratio', 'stoch_k_14_3_3', 'vwap_ratio', 'price_action'],
        '1h': ['open', 'high', 'low', 'close', 'volume', 'rsi_21',
               'macd_21_42_9',
               'bb_width_20_2', 'adx_14', 'obv_ratio_20', 'ema_50_ratio',
               'ichimoku_base', 'fib_ratio', 'price_ema_ratio_50'],
        '4h': ['open', 'high', 'low', 'close', 'volume', 'rsi_28',
               'macd_26_52_18',
               'supertrend_10_3', 'volume_sma_20_ratio', 'ema_100_ratio',
               'pivot_level', 'donchian_width_20', 'market_structure',
               'volatility_ratio_14_50']
    }
    
    for tf in timeframes:
        parquet_file = data_dir / f'{tf}.parquet'
        
        if not parquet_file.exists():
            print(f"❌ File not found: {parquet_file}")
            continue
        
        print(f"\n📊 Loading {tf} training data from {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        print(f"   Shape before filtering: {df.shape}")
        
        # Filter to training columns (same as in feature_engineer.py)
        cols_to_keep = [c for c in train_columns[tf] if c in df.columns]
        df = df[cols_to_keep]
        print(f"   Shape after filtering: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Fit scaler
        scaler = scaler_types[tf]
        print(f"   Fitting {scaler.__class__.__name__}...")
        scaler.fit(df.values)
        
        scalers[tf] = scaler
        print(f"   ✅ {tf} scaler fitted: {scaler.n_features_in_} features")
    
    # Save scalers
    print(f"\n💾 Saving scalers to {output_path}...")
    joblib.dump(scalers, output_path)
    print(f"✅ Scalers saved successfully!")
    
    # Verify
    print(f"\n🔍 Verifying saved scalers...")
    loaded_scalers = joblib.load(output_path)
    for tf, scaler in loaded_scalers.items():
        print(f"   ✅ {tf}: {scaler.__class__.__name__} ({scaler.n_features_in_} features)")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING SCALERS GENERATED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    generate_scalers()
