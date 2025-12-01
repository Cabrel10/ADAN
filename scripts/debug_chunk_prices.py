#!/usr/bin/env python
"""
Debug script: Check actual chunk loading and price extraction
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

config_loader = ConfigLoader()
config = config_loader.load_config("config/config.yaml")

def make_env():
    return RealisticTradingEnv(
        config=config,
        worker_config=config["workers"]["w2"],
        worker_id=0,
        enable_market_friction=False,
        use_stable_reward=False
    )

env = DummyVecEnv([make_env])
real_env = env.envs[0]

print("DEBUGGING ENV AFTER RESET")
print("="*60)
obs = env.reset()

print(f"step_in_chunk: {real_env.step_in_chunk}")
print(f"current_data keys: {list(real_env.current_data.keys()) if hasattr(real_env, 'current_data') and real_env.current_data else 'NONE'}")

if hasattr(real_env, 'current_data') and real_env.current_data:
    for asset, tfs in real_env.current_data.items():
        print(f"\n{asset}:")
        for tf, df in tfs.items():
            if df is not None:
                print(f"  {tf}: {len(df)} rows, step_in_chunk={real_env.step_in_chunk}")
                if real_env.step_in_chunk < len(df):
                    close_price = df.iloc[real_env.step_in_chunk].get('close', 'NO CLOSE COL')
                    print(f"    close at step {real_env.step_in_chunk}: {close_price}")
                else:
                    print(f"    ❌ step_in_chunk OUT OF BOUNDS! ({real_env.step_in_chunk} >= {len(df)})")

print(f"\nCalling _get_current_prices()...")
prices = real_env._get_current_prices()
print(f"Prices returned: {prices}")
