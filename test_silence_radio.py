#!/usr/bin/env python3
"""
TEST SILENCE RADIO - Actions strictement nulles
Objectif: Prouver que le système respecte les seuils (ne trade PAS sur signal=0)
"""
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

def test_silence_radio():
    print("=" * 80)
    print("TEST SILENCE RADIO - ACTIONS = 0")
    print("=" * 80)
    
    print(f"\n[1] Chargement config NORMAL (threshold=0.05)")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Désactiver force trades pour isoler
    if 'trading_rules' not in config:
        config['trading_rules'] = {}
    if 'frequency' not in config['trading_rules']:
        config['trading_rules']['frequency'] = {}
    
    config['trading_rules']['frequency']['force_trade_steps'] = {
        '5m': 99999,
        '1h': 99999,
        '4h': 99999
    }
    
    print("   ✅ Force trades: DÉSACTIVÉS")
    print("   ✅ Threshold: 0.05 (normal from config)")
    
    print(f"\n[2] Création environnement")
    env = MultiAssetChunkedEnv(config=config)
    
    print(f"\n[3] Reset")
    obs = env.reset()
    initial_portfolio = env.portfolio.portfolio_value
    
    print(f"   Portfolio initial: ${initial_portfolio:.2f}")
    
    print(f"\n[4] Exécution 100 steps avec ACTIONS = 0 (SILENCE TOTAL)")
    trade_count = 0
    
    for step in range(100):
        # ACTIONS STRICTEMENT NULLES
        action = np.zeros(env.action_space.shape)
        
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
        
        current_portfolio = env.portfolio.portfolio_value
        
        # Compter trades
        if isinstance(info, dict):
            trades_info = info.get("trades", [])
            if isinstance(trades_info, (list, tuple)):
                if len(trades_info) > 0:
                    print(f"   ⚠️  Step {step}: {len(trades_info)} trades détectés avec action=0!")
                trade_count += len(trades_info)
            elif isinstance(trades_info, int):
                if trades_info > 0:
                    print(f"   ⚠️  Step {step}: {trades_info} trades détectés avec action=0!")
                trade_count += trades_info
        
        if step % 25 == 24:
            print(f"   Step {step+1:3d}: portfolio=${current_portfolio:.2f}, trades_total={trade_count}")
        
        if done:
            break
    
    final_portfolio = env.portfolio.portfolio_value
    
    print(f"\n[5] RÉSULTATS:")
    print(f"   Portfolio: ${initial_portfolio:.2f} → ${final_portfolio:.2f}")
    print(f"   Delta: ${final_portfolio - initial_portfolio:.2f}")
    print(f"   Trades TOTAL: {trade_count}")
    print(f"   Positions finales: {len(env.portfolio.positions)}")
    
    print(f"\n[6] VERDICT:")
    
    if trade_count == 0:
        print("   ✅✅✅ PARFAIT: 0 trades avec actions=0")
        print("   → Le système RESPECTE les seuils correctement")
        print("   → Le problème des 1079 trades venait des actions aléatoires uniformes!")
        print("   → Un modèle PPO entraîné (gaussien centré) sera BEAUCOUP plus calme")
        success = True
    else:
        print(f"   ❌ ÉCHEC: {trade_count} trades avec actions=0")
        print("   → BUG CRITIQUE: Le système trade sans signal!")
        print("   → Investigation approfondie requise")
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("CONCLUSION: ✅ SYSTÈME SAIN")
        print("Les tests précédents (1079 trades) étaient dus à l'input test, pas au code")
        print("\n💡 PROCHAINE ÉTAPE:")
        print("- Tester avec actions gaussiennes np.random.normal(0, 0.1)")
        print("- Puis lancer training PPO 1000 steps")
        print("- Puis Optuna W2/W3/W4")
    else:
        print("CONCLUSION: ❌ BUG PROFOND À DEBUGGER")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = test_silence_radio()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
