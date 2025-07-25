#!/usr/bin/env python3

import sys
import os
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Load real configuration from YAML
with open('config/environment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Add required data configuration for the test
config["data"] = {
    "data_dir": "data/final",
    "assets": ["BTC"],
    "chunk_size": 1000
}

print("=== Features Config Debug ===")

# Create environment instance to test _get_features_config
env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
env.config = config

# Test the _get_features_config method
features_config = env._get_features_config()
print(f"Features config: {features_config}")

for tf, features in features_config.items():
    if 'minutes_since_update' in features:
        print(f"✅ {tf}: minutes_since_update is included")
    else:
        print(f"❌ {tf}: minutes_since_update is MISSING")