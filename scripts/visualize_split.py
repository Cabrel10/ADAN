#!/usr/bin/env python3
"""
VISUALIZE DATA SPLIT
Verifies the strict separation between Train and Test sets visually.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_split(asset="BTCUSDT", timeframe="1h"):
    train_path = Path(f"data/processed/indicators/train/{asset}/{timeframe}.parquet")
    test_path = Path(f"data/processed/indicators/test/{asset}/{timeframe}.parquet")
    
    if not train_path.exists() or not test_path.exists():
        logger.error(f"❌ Data not found for {asset} {timeframe}")
        return
    
    logger.info(f"Loading data for {asset} {timeframe}...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    plt.figure(figsize=(15, 7))
    plt.plot(df_train.index, df_train['close'], label='Train (2021-2023)', color='blue', alpha=0.7)
    plt.plot(df_test.index, df_test['close'], label='Test (2024-Now)', color='orange', alpha=0.7)
    
    plt.title(f"{asset} Data Split Verification ({timeframe})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"data_split_{asset}_{timeframe}.png"
    
    plt.savefig(output_path)
    logger.info(f"✅ Saved plot to {output_path}")
    
    # Print stats
    logger.info(f"Train End: {df_train.index.max()}")
    logger.info(f"Test Start: {df_test.index.min()}")
    gap = df_test.index.min() - df_train.index.max()
    logger.info(f"Gap: {gap}")

if __name__ == "__main__":
    visualize_split("BTCUSDT", "1h")
    visualize_split("XRPUSDT", "1h")
