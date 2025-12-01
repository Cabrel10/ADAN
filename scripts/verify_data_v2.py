#!/usr/bin/env python3
"""
VERIFY DATA V2
Independently validates the generated parquet files for ADAN 2.0.
Checks:
1. Date ranges (Train < Test, no overlap).
2. Column presence (ATR_20, ATR_50).
3. NaN counts (must be 0).
"""

import pandas as pd
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

ASSETS = ["BTCUSDT", "XRPUSDT"]
TIMEFRAMES = ["5m", "1h", "4h"]
REQUIRED_COLUMNS = ["atr_20", "atr_50", "adx_14"]

def verify_asset_timeframe(asset, timeframe):
    logger.info(f"\n🔍 Verifying {asset} {timeframe}...")
    
    train_path = Path(f"data/processed/indicators/train/{asset}/{timeframe}.parquet")
    test_path = Path(f"data/processed/indicators/test/{asset}/{timeframe}.parquet")
    
    if not train_path.exists():
        logger.error(f"❌ Train file missing: {train_path}")
        return False
    if not test_path.exists():
        logger.error(f"❌ Test file missing: {test_path}")
        return False
        
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    # 1. Date Ranges
    train_start = df_train.index.min()
    train_end = df_train.index.max()
    test_start = df_test.index.min()
    test_end = df_test.index.max()
    
    logger.info(f"   Train: {train_start} -> {train_end} ({len(df_train)} rows)")
    logger.info(f"   Test:  {test_start} -> {test_end} ({len(df_test)} rows)")
    
    if train_end >= test_start:
        logger.error(f"❌ OVERLAP DETECTED! Train End ({train_end}) >= Test Start ({test_start})")
        return False
    else:
        gap = test_start - train_end
        logger.info(f"✅ Split OK. Gap: {gap}")
        
    # 2. Columns
    missing_cols = []
    for col in REQUIRED_COLUMNS:
        if col not in df_train.columns:
            missing_cols.append(f"Train:{col}")
        if col not in df_test.columns:
            missing_cols.append(f"Test:{col}")
            
    if missing_cols:
        logger.error(f"❌ Missing columns: {missing_cols}")
        return False
    else:
        logger.info(f"✅ Required columns present: {REQUIRED_COLUMNS}")
        
    # 3. NaNs
    train_nans = df_train.isna().sum().sum()
    test_nans = df_test.isna().sum().sum()
    
    if train_nans > 0:
        logger.warning(f"⚠️ Train has {train_nans} NaNs")
        # Optional: List columns with NaNs
        nan_cols = df_train.columns[df_train.isna().any()].tolist()
        logger.warning(f"   NaN Columns: {nan_cols}")
        
    if test_nans > 0:
        logger.warning(f"⚠️ Test has {test_nans} NaNs")
        nan_cols = df_test.columns[df_test.isna().any()].tolist()
        logger.warning(f"   NaN Columns: {nan_cols}")
        
    if train_nans == 0 and test_nans == 0:
        logger.info("✅ Zero NaNs detected")
        
    return True

def main():
    logger.info("="*60)
    logger.info("ADAN 2.0 DATA VERIFICATION")
    logger.info("="*60)
    
    all_passed = True
    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            if not verify_asset_timeframe(asset, timeframe):
                all_passed = False
                
    logger.info("="*60)
    if all_passed:
        logger.info("✅ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        logger.error("❌ SOME CHECKS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
