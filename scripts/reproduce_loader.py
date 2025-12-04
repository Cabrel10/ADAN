#!/usr/bin/env python3
"""
Reproduction script to verify DataLoader behavior and 0.0 values.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.common.config_loader import ConfigLoader

def test_loader():
    print("="*80)
    print("TESTING DATA LOADER")
    print("="*80)

    # Mock config
    config = {
        "data": {
            "data_split": "train",
            "chunk_sizes": {"5m": 5328, "1h": 242, "4h": 111},
            "features_config": {
                "timeframes": {
                    "5m": {"price": "close"},
                    "1h": {"price": "close"},
                    "4h": {"price": "close"}
                }
            },
            "data_dirs": {
                "train": "data/processed/indicators/train"
            }
        },
        "paths": {
            "processed_data_dir": "data/processed"
        },
        "environment": {
            "assets": ["BTCUSDT"]
        }
    }

    worker_config = {
        "timeframes": ["5m", "1h", "4h"],
        "assets": ["BTCUSDT"],
        "data_split_override": "train"
    }

    try:
        loader = ChunkedDataLoader(config, worker_config, worker_id=0)
        
        # Load chunk 1 (as seen in user logs)
        print("\nLoading chunk 1...")
        data = loader.load_chunk(1)
        
        btc_data = data.get("BTCUSDT", {})
        df_4h = btc_data.get("4h")
        
        if df_4h is None:
            print("ERROR: 4h DataFrame is None!")
            return

        print(f"\n4h DataFrame Shape: {df_4h.shape}")
        print(f"Columns: {list(df_4h.columns)}")
        
        # Find close column
        close_col = next((c for c in df_4h.columns if 'close' in c.lower()), None)
        print(f"Close column: {close_col}")
        
        if close_col:
            # Check for 0.0
            zeros = df_4h[df_4h[close_col] <= 0]
            print(f"\nRows with {close_col} <= 0.0: {len(zeros)}")
            if len(zeros) > 0:
                print(zeros)
            
            # Check last rows
            print(f"\nLast 5 rows:")
            print(df_4h[[close_col]].tail())
            
            # Check specific indices if size is 111
            if len(df_4h) == 111:
                print(f"\nIndex 110 value: {df_4h.iloc[110][close_col]}")
            
            # Simulate step_idx calculation
            print(f"\n{'='*80}")
            print("SIMULATING INDEXING")
            print(f"{'='*80}")
            
            # Scenario 1: Start of chunk
            base_step = 0
            ratio = 48.0
            idx = int(base_step / ratio)
            print(f"Base Step {base_step} (5m) -> 4h Index {idx}")
            print(f"Value: {df_4h.iloc[idx][close_col]}")
            
            # Scenario 2: End of chunk (assuming 5m chunk size 5328)
            base_step = 5327
            idx = int(base_step / ratio)
            print(f"Base Step {base_step} (5m) -> 4h Index {idx}")
            if idx < len(df_4h):
                print(f"Value: {df_4h.iloc[idx][close_col]}")
            else:
                print(f"Index {idx} out of bounds (Size {len(df_4h)})")
                
        # Check 1h Data
        print(f"\n{'='*80}")
        print("CHECKING 1H DATA")
        print(f"{'='*80}")
        
        df_1h = btc_data.get("1h")
        if df_1h is None:
            print("ERROR: 1h DataFrame is None!")
        else:
            print(f"1h DataFrame Shape: {df_1h.shape}")
            close_col_1h = next((c for c in df_1h.columns if 'close' in c.lower()), None)
            print(f"Close column: {close_col_1h}")
            
            if close_col_1h:
                print("\nFirst 20 rows of 1h data:")
                print(df_1h[[close_col_1h]].head(20))
                
                # Check for 0.0
                zeros_1h = df_1h[df_1h[close_col_1h] <= 0]
                print(f"\nRows with {close_col_1h} <= 0.0: {len(zeros_1h)}")
                if len(zeros_1h) > 0:
                    print(zeros_1h[[close_col_1h]])
                    
                # Check index 7 specifically
                if len(df_1h) > 7:
                    print(f"\nIndex 7 value: {df_1h.iloc[7][close_col_1h]}")

    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loader()
