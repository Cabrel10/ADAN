#!/usr/bin/env python3
"""
TEST HONNÊTE - Sweep Threshold
Objectif: Tester RÉELLEMENT chaque threshold, pas des stats théoriques
"""
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

def test_threshold(threshold_value):
    """Test un threshold spécifique et compte VRAIMENT les trades"""
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Modifier le threshold
    config['environment']['action_thresholds']['5m'] = threshold_value
    config['environment']['action_thresholds']['1h'] = threshold_value
    config['environment']['action_thresholds']['4h'] = threshold_value
    
    # Désactiver force trades pour isoler
    config['trading_rules']['frequency']['force_trade_steps'] = {
        '5m': 99999, '1h': 99999, '4h': 99999
    }
    
    env = MultiAssetChunkedEnv(config=config)
    obs = env.reset()
    
    np.random.seed(42)  # Reproductible
    trades_total = 0
    max_action_seen = 0
    
    for i in range(50):  # Test court
        action = np.random.normal(0, 0.1, 25)
        max_abs_action = np.max(np.abs(action))
        max_action_seen = max(max_action_seen, max_abs_action)
        
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
        
        if isinstance(info, dict):
            trades_this_step = info.get('executed_trades_opened', 0)
            trades_total += trades_this_step
            
            if trades_this_step > 0 and max_abs_action < threshold_value:
                print(f"   ⚠️  STEP {i}: TRADED avec action={max_abs_action:.3f} < threshold={threshold_value}")
    
    return trades_total, max_action_seen

if __name__ == "__main__":
    print("=" * 80)
    print("TEST HONNÊTE - SWEEP THRESHOLD")
    print("=" * 80)
    
    print("\nTest sur 50 steps avec actions gaussiennes(0, 0.1)")
    print("Force trades DÉSACTIVÉS pour isoler natural trades\n")
    
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.99]
    results = []
    
    for th in thresholds:
        trades, max_action = test_threshold(th)
        results.append((th, trades, max_action))
        status = "✅" if trades < 10 else ("🟡" if trades < 30 else "❌")
        print(f"{status} Threshold {th:.2f}: {trades:3d} trades (max|action|={max_action:.3f})")
    
    print("\n" + "=" * 80)
    print("ANALYSE:")
    
    # Trouver le point où ça devient raisonnable
    reasonable = [r for r in results if r[1] < 15]
    if reasonable:
        optimal = reasonable[0]
        print(f"✅ Threshold optimal: {optimal[0]:.2f} ({optimal[1]} trades)")
    else:
        print(f"❌ AUCUN threshold donne <15 trades → BUG DE LOGIQUE")
    
    # Vérifier bug
    bug_found = False
    for th, trades, max_action in results:
        if max_action < th and trades > 0:
            print(f"🔴 BUG CONFIRMÉ: Threshold {th:.2f}, max action {max_action:.3f}, mais {trades} trades!")
            bug_found = True
    
    if not bug_found:
        print("✅ Pas de bug évident (trades seulement quand action > threshold)")
    
    print("=" * 80)
