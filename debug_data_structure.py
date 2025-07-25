#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from adan_trading_bot.data_processing.chunked_loader import ChunkedDataLoader
from pathlib import Path

# Initialize the data loader
loader = ChunkedDataLoader(
    data_dir=Path("data/final"),
    chunk_size=1000,
    assets_list=["BTC", "ETH"],
    split="train"
)

print("=== Data Structure Debug ===")

# Load first chunk
chunk_data = loader.load_chunk(0)
print(f"Chunk data keys (assets): {list(chunk_data.keys())}")

for asset, timeframe_data in chunk_data.items():
    print(f"\nAsset: {asset}")
    print(f"  Timeframes: {list(timeframe_data.keys())}")
    
    for tf, df in timeframe_data.items():
        print(f"  Timeframe {tf}:")
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)[:10]}...")  # First 10 columns
        break  # Only show first timeframe for brevity
    break  # Only show first asset for brevity