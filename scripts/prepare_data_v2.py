#!/usr/bin/env python3
"""
ADAN 2.0 DATA PREPARATION SCRIPT
Strict separation of Train (2021-2023) and Test (2024-Now) data.
Calculates all indicators including ATR_20, ATR_50.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer

# Constants
ASSETS = ["BTCUSDT", "XRPUSDT"]
TIMEFRAMES = ["5m", "1h", "4h"]
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
# Test end is "Now", we'll fetch up to current time

def download_data(asset: str, start: str, end: str = None, timeframe: str = "5m", retries: int = 3):
    """
    Downloads OHLCV data using ccxt with retry logic.
    """
    logger.info(f"Downloading {asset} {timeframe} from {start} to {end if end else 'Now'}...")
    
    import ccxt
    import time
    
    for attempt in range(retries):
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}  # Use futures data if possible, or spot
            })
            # Try spot if future fails or just default to spot? 
            # Let's stick to default which is usually spot, but maybe we want futures?
            # The user didn't specify, but usually bots trade futures. 
            # Let's keep it simple first: default binance (spot).
            exchange = ccxt.binance({'enableRateLimit': True})
            
            start_dt = pd.to_datetime(start)
            if end:
                end_dt = pd.to_datetime(end)
            else:
                end_dt = datetime.now()
                
            all_data = []
            current = start_dt
            
            while current < end_dt:
                try:
                    since = int(current.timestamp() * 1000)
                    ohlcv = exchange.fetch_ohlcv(asset, timeframe, since=since, limit=1000)
                    
                    if not ohlcv:
                        break
                    
                    df_chunk = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                    
                    # Verify we are getting new data
                    if not all_data and df_chunk.empty:
                        break
                        
                    all_data.append(df_chunk)
                    
                    last_time = df_chunk['timestamp'].max()
                    if last_time <= current:
                        # Stuck in loop
                        break
                        
                    current = last_time + timedelta(minutes=1) # Small increment to avoid overlap/gap issues handled by ccxt usually
                    
                    # Simple progress log
                    if len(all_data) % 10 == 0:
                        logger.info(f"  Downloaded up to {current}")
                    
                except Exception as e:
                    logger.warning(f"  Error during download chunk (Attempt {attempt+1}): {e}")
                    time.sleep(2) # Wait a bit
                    # Don't break, try next chunk? Or retry entire download?
                    # Usually better to retry the chunk. But here we are in a loop.
                    # Let's retry the whole download if it fails significantly.
                    raise e 
            
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp'])
                df = df.sort_values('timestamp')
                df.set_index('timestamp', inplace=True)
                
                # Filter exactly to range
                mask = (df.index >= start_dt)
                if end:
                    mask &= (df.index <= end_dt)
                df = df[mask]
                
                logger.info(f"✅ Downloaded {len(df)} candles for {asset}")
                return df
            else:
                logger.warning(f"⚠️ No data found for {asset} on attempt {attempt+1}")
                
        except Exception as e:
            logger.error(f"❌ Download failed (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(5) # Backoff
            
    logger.error(f"❌ Failed to download {asset} after {retries} attempts")
    return None

def process_and_save(df: pd.DataFrame, asset: str, config: dict, mode: str):
    """
    Calculates indicators and saves to parquet.
    mode: 'train' or 'test'
    """
    base_dir = Path(f"data/processed/indicators/{mode}/{asset}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # We need to resample for higher timeframes from the 5m base data
    # OR download them separately. Downloading separately is safer for consistency with exchange.
    # But here we downloaded 5m. Let's download 1h and 4h separately to be precise.
    # Actually, the requirement implies we should have data for all timeframes.
    
    # Optimization: To ensure perfect alignment, we should download 1h and 4h data separately
    # rather than resampling, as exchange data is the ground truth.
    
    return results

def main():
    logger.info("=" * 80)
    logger.info("ADAN 2.0 DATA PREPARATION")
    logger.info(f"Train: {TRAIN_START} -> {TRAIN_END}")
    logger.info(f"Test:  {TEST_START} -> Now")
    logger.info("=" * 80)
    
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    
    for asset in ASSETS:
        logger.info(f"\nProcessing {asset}...")
        
        for timeframe in TIMEFRAMES:
            # 1. Download Full History (start of Train to Now)
            # We download everything first to ensure continuity if we needed to resample,
            # but here we will download specific ranges to be efficient or just filter.
            # Let's download the full range to ensure we handle the boundary correctly.
            
            # Actually, let's download Train and Test separately to be explicit.
            
            # --- TRAIN SET ---
            logger.info(f"  [TRAIN] Fetching {timeframe}...")
            df_train = download_data(asset, TRAIN_START, TRAIN_END, timeframe)
            
            if df_train is not None and not df_train.empty:
                # Calculate Indicators
                try:
                    fe = FeatureEngineer(config, ".")
                    df_train_ind = fe.calculate_indicators_for_single_timeframe(df_train, timeframe)
                    
                    # Save
                    save_path = Path(f"data/processed/indicators/train/{asset}/{timeframe}.parquet")
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    df_train_ind.to_parquet(save_path)
                    logger.info(f"  ✅ Saved TRAIN to {save_path}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Failed to process TRAIN {timeframe}: {e}")
            
            # --- TEST SET ---
            logger.info(f"  [TEST] Fetching {timeframe}...")
            df_test = download_data(asset, TEST_START, end=None, timeframe=timeframe)
            
            if df_test is not None and not df_test.empty:
                # Calculate Indicators
                try:
                    fe = FeatureEngineer(config, ".")
                    df_test_ind = fe.calculate_indicators_for_single_timeframe(df_test, timeframe)
                    
                    # Save
                    save_path = Path(f"data/processed/indicators/test/{asset}/{timeframe}.parquet")
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    df_test_ind.to_parquet(save_path)
                    logger.info(f"  ✅ Saved TEST to {save_path}")
                    
                    # VALIDATION
                    if df_train is not None:
                        train_max = df_train.index.max()
                        test_min = df_test.index.min()
                        if train_max >= test_min:
                            logger.error(f"  ❌ OVERLAP DETECTED! Train Max: {train_max}, Test Min: {test_min}")
                        else:
                            logger.info(f"  ✅ No Overlap. Gap: {test_min - train_max}")
                            
                    # Check for NaNs
                    nan_count = df_test_ind.isna().sum().sum()
                    if nan_count > 0:
                         logger.warning(f"  ⚠️ {nan_count} NaNs in Test Set!")
                    else:
                         logger.info("  ✅ Zero NaNs in Test Set")

                except Exception as e:
                    logger.error(f"  ❌ Failed to process TEST {timeframe}: {e}")

if __name__ == "__main__":
    main()
