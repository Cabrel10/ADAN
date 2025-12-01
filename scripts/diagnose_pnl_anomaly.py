#!/usr/bin/env python
"""
Diagnostic: PnL/Trades Mismatch Investigation
Verifies if PnL comes from unrealized (open) positions
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

print("="*60)
print("DIAGNOSTIC: PnL/Trades Anomaly Investigation")
print("="*60)

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

# Load VecNormalize
try:
    env = VecNormalize.load("models/rl_agents/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False
except:
    pass

# Load model
model = PPO.load("models/rl_agents/final/w2_final.zip", env=env)

print("\nRunning 100 steps episode...")
obs = env.reset()
episode_reward = 0

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward[0]
    
    if done[0]:
        break

print("\n" + "="*60)
print("EPISODE END - DIAGNOSTICS")
print("="*60)

real_env = env.envs[0]

# Get portfolio state
portfolio = real_env.portfolio_manager if hasattr(real_env, 'portfolio_manager') else None

if portfolio:
    print(f"\n📊 PORTFOLIO STATE:")
    print(f"  Balance: ${portfolio.balance:.2f}")
    print(f"  Total equity: ${portfolio.get_total_equity():.2f}")
    
    # Check open positions
    open_positions = portfolio.get_open_positions() if hasattr(portfolio, 'get_open_positions') else {}
    print(f"\n📈 OPEN POSITIONS: {len(open_positions)}")
    
    if open_positions:
        for asset, pos in open_positions.items():
            print(f"  {asset}: {pos.get('quantity', 0)} units @ ${pos.get('entry_price', 0):.2f}")
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            print(f"    Unrealized PnL: ${unrealized_pnl:.2f}")
    
    # Check trade history
    if hasattr(portfolio, 'trade_history'):
        closed_trades = len(portfolio.trade_history)
        print(f"\n📝 CLOSED TRADES: {closed_trades}")
        
        if portfolio.trade_history:
            total_realized_pnl = sum(t.get('pnl', 0) for t in portfolio.trade_history)
            print(f"  Total realized PnL: ${total_realized_pnl:.2f}")
    
    # Performance metrics
    if hasattr(real_env, 'performance_metrics') and real_env.performance_metrics:
        metrics = real_env.performance_metrics.get_metrics_summary()
        print(f"\n📊 METRICS SUMMARY:")
        print(f"  Daily trades: {metrics.get('daily_trades', 0)}")
        print(f"  Daily PnL: ${metrics.get('daily_pnl', 0):.2f}")
        print(f"  Win rate: {metrics.get('win_rate', 0):.2%}")
else:
    print("⚠️ Portfolio manager not accessible")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if portfolio and open_positions:
    print("✅ HYPOTHESIS CONFIRMED: Unrealized PnL from open positions")
    print(f"   {len(open_positions)} position(s) still open at episode end")
    print("\n🔧 RECOMMENDED FIX:")
    print("   Add close_all_positions() in episode termination")
else:
    print("❌ No open positions found - need deeper investigation")

print("="*60)
