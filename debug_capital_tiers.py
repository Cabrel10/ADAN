#!/usr/bin/env python3

import sys
import os
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
import pandas as pd

# Load real configuration from YAML
with open('config/environment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Add required data configuration for the test
config["data"] = {
    "data_dir": "data/final",
    "assets": ["ASSET_1", "ASSET_2"],
    "chunk_size": 1000
}

# Add portfolio config if not present
if "portfolio" not in config:
    config["portfolio"] = {"initial_balance": 10000.0}

print("Testing capital_tiers conversion in MultiAssetChunkedEnv...")
print(f"Original capital_tiers: {config['capital_tiers']}")

# Create some dummy data files for the test
os.makedirs("data/final/ASSET_1", exist_ok=True)
os.makedirs("data/final/ASSET_2", exist_ok=True)

# Create minimal test data
test_data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=10, freq='5min'),
    'open': [100.0] * 10,
    'high': [101.0] * 10,
    'low': [99.0] * 10,
    'close': [100.5] * 10,
    'volume': [1000.0] * 10
})

test_data.to_parquet("data/final/ASSET_1/train.parquet")
test_data.to_parquet("data/final/ASSET_2/train.parquet")

try:
    # This should trigger the capital_tiers conversion
    env = MultiAssetChunkedEnv(config)
    print("MultiAssetChunkedEnv created successfully!")
    print(f"Converted capital_tiers: {env.portfolio.capital_tiers}")
    print(f"capital_tiers type: {type(env.portfolio.capital_tiers)}")
    
    # Test get_current_tier
    tier = env.portfolio.get_current_tier()
    print(f"Current tier: {tier}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()