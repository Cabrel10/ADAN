#!/usr/bin/env python3
"""
TEST PHASE 3 - GAUSSIEN (PPO-LIKE)
Objectif: Simuler un modèle PPO initialisé avec actions réalistes
Attendu: 5-20 trades en 100 steps (ni 0, ni 1000)
"""
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

def test_gaussian():
    print("=" * 80)
    print("TEST PHASE 3 - ACTIONS GAUSSIENNES (PPO-LIKE)")
    print("=" * 80)
    
    print(f"\n[1] Configuration")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Garder force trades avec espacement normal (pas 99999)
    # pour simuler environnement réel
    print("   Config standard (force_trade espacé normalement)")
    
    print(f"\n[2] Initialisation")
    env = MultiAssetChunkedEnv(config=config)
    obs = env.reset()
    
    initial_portfolio = env.portfolio.portfolio_value
    print(f"   Portfolio initial: ${initial_portfolio:.2f}")
    
    print(f"\n[3] Simulation PPO (100 steps)")
    print("   Actions: Gaussienne(μ=0, σ=0.1)")
    
    np.random.seed(42)  # Reproductibilité
    
    trade_count = 0
    portfolio_history = [initial_portfolio]
    force_trades = 0
    natural_trades = 0
    
    for step in range(100):
        # Actions gaussiennes (comme PPO initialisé)
        # Centré sur 0, std=0.1
        action = np.random.normal(0, 0.1, env.action_space.shape)
        
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
        
        current_portfolio = env.portfolio.portfolio_value
        portfolio_history.append(current_portfolio)
        
        # Compter trades
        if isinstance(info, dict):
            trades_info = info.get("trades", [])
            if isinstance(trades_info, (list, tuple)):
                step_trades = len(trades_info)
            elif isinstance(trades_info, int):
                step_trades = trades_info
            else:
                step_trades = 1 if trades_info else 0
            
            trade_count += step_trades
            
            # Distinguer force vs natural (approximatif)
            # Si info contient 'force_trade' ou step est multiple de force_trade_steps
            if step_trades > 0:
                # Logs montreraient si c'est force ou natural
                pass
        
        if step % 20 == 19:
            max_action = np.max(np.abs(action))
            print(f"   Step {step+1:3d}: portfolio=${current_portfolio:.2f}, "
                  f"trades_total={trade_count}, max|action|={max_action:.3f}")
        
        if done:
            break
    
    final_portfolio = env.portfolio.portfolio_value
    portfolio_variance = np.std(portfolio_history)
    total_return = (final_portfolio / initial_portfolio - 1) * 100
    
    print(f"\n[4] RÉSULTATS:")
    print(f"   Portfolio: ${initial_portfolio:.2f} → ${final_portfolio:.2f}")
    print(f"   Return: {total_return:.2f}%")
    print(f"   Variance: ${portfolio_variance:.2f}")
    print(f"   Trades TOTAL: {trade_count}")
    print(f"   Trades/step: {trade_count/100:.2f}")
    
    # Analyse distribution actions
    print(f"\n[5] ANALYSE ACTIONS:")
    sample_actions = [np.random.normal(0, 0.1, env.action_space.shape) for _ in range(1000)]
    sample_max = [np.max(np.abs(a)) for a in sample_actions]
    pct_above_005 = np.mean([m > 0.05 for m in sample_max]) * 100
    pct_above_010 = np.mean([m > 0.10 for m in sample_max]) * 100
    
    print(f"   P(max|action| > 0.05) = {pct_above_005:.1f}%")
    print(f"   P(max|action| > 0.10) = {pct_above_010:.1f}%")
    print(f"   → Attendu théorique: ~{pct_above_005:.0f} trades/100 steps")
    
    print(f"\n[6] VERDICT:")
    
    success = True
    
    if trade_count == 0:
        print("   ❌ ÉCHEC: 0 trades (système trop restrictif)")
        success = False
    elif trade_count < 5:
        print(f"   ⚠️  FAIBLE: {trade_count} trades (peut-être trop sélectif)")
        print("   Acceptable si force trades désactivés")
    elif 5 <= trade_count <= 30:
        print(f"   ✅ PARFAIT: {trade_count} trades (réaliste pour PPO)")
        print("   Fréquence adaptée pour apprentissage")
    elif 30 < trade_count <= 100:
        print(f"   🟡 ÉLEVÉ: {trade_count} trades (un peu actif)")
        print("   Acceptable, mais peut nécessiter ajustement threshold")
    else:
        print(f"   ❌ OVER-TRADING: {trade_count} trades (toujours problématique)")
        print("   Threshold ou logique encore buggée")
        success = False
    
    if portfolio_variance < 0.10:
        print(f"   ⚠️  Portfolio très stable (var={portfolio_variance:.2f})")
        if trade_count > 0:
            print("   Trades existent mais impactent peu → vérifier sizes")
    
    print("\n" + "=" * 80)
    if success and 5 <= trade_count <= 30:
        print("CONCLUSION: ✅ ENVIRONNEMENT PRÊT POUR PPO TRAINING")
        print("\n💡 PROCHAINE ÉTAPE:")
        print("   Phase 4: Training PPO 1000-2000 steps")
        print("   - Vérifier convergence (loss diminue)")
        print("   - Vérifier apprentissage (Sharpe augmente)")
        print("   Puis: Optuna W2/W3/W4 (20 trials chacun)")
    else:
        print("CONCLUSION: 🟡 SYSTÈME FONCTIONNEL MAIS AJUSTEMENTS POSSIBLES")
        if trade_count > 30:
            print("   Recommandation: Augmenter threshold à 0.08-0.10")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = test_gaussian()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
