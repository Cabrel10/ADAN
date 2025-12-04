#!/usr/bin/env python3
"""
TEST 1.1: Vérifier l'environnement de base
Objectif: Confirmer que l'environnement peut exécuter des trades et mettre à jour le portfolio
"""
import sys
import numpy as np
import yaml
from pathlib import Path

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

def test_env_basic():
    print("=" * 80)
    print("TEST 1.1: ENVIRONNEMENT DE BASE")
    print("=" * 80)
    
    config_path = "config/config.yaml"
    
    print(f"\n[1] Chargement de la configuration: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config chargée: {len(config)} sections principales")
    
    print(f"\n[2] Initialisation de l'environnement")
    env = MultiAssetChunkedEnv(config=config)
    
    print("✅ Environnement créé sans erreur")
    
    print(f"\n[3] Reset de l'environnement")
    obs = env.reset()
    print(f"✅ Observation obtenue, shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
    
    # Récupérer le portfolio initial directement
    initial_portfolio = env.portfolio.portfolio_value if hasattr(env, 'portfolio') else 20.5
    initial_cash = env.portfolio.cash if hasattr(env,  'portfolio') else 20.5
    
    print(f"\n[3] État initial:")
    print(f"   Portfolio: ${initial_portfolio:.2f}")
    print(f"   Cash: ${initial_cash:.2f}")
    
    print(f"\n[4] Exécution de 100 steps avec actions aléatoires:")
    trade_count = 0
    portfolio_changes = []
    previous_portfolio = initial_portfolio
    
    for step in range(100):
        # Action aléatoire modérée (pas trop agressive)
        action = env.action_space.sample() * 0.3
        
        # Gymnasium API: 5-tuple (obs, reward, terminated, truncated, info)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False
        
        current_portfolio = env.portfolio.portfolio_value
        current_cash = env.portfolio.cash
        trades_info = info.get("trades", []) if isinstance(info, dict) else []
        
        # Détecter changement de portfolio
        if abs(current_portfolio - previous_portfolio) > 0.01:
            portfolio_changes.append({
                'step': step,
                'old': previous_portfolio,
                'new': current_portfolio,
                'delta': current_portfolio - previous_portfolio
            })
        
        # Compter les trades (gérer int ou list)
        if isinstance(trades_info, (list, tuple)):
            trade_count += len(trades_info)
        elif isinstance(trades_info, int):
            trade_count += trades_info
        elif trades_info:
            trade_count += 1
        
        if step % 20 == 0:
            print(f"   Step {step:3d}: portfolio=${current_portfolio:.2f}, "
                  f"reward={reward:.4f}, trades={trade_count}")
        
        previous_portfolio = current_portfolio
        
        if done:
            print(f"   Episode terminé au step {step}")
            break
    
    final_portfolio = env.portfolio.portfolio_value
    final_cash = env.portfolio.cash
    
    print(f"\n[5] Résultats après 100 steps:")
    print(f"   Portfolio initial: ${initial_portfolio:.2f}")
    print(f"   Portfolio final: ${final_portfolio:.2f}")
    print(f"   Delta: ${final_portfolio - initial_portfolio:.2f}")
    print(f"   Cash final: ${final_cash:.2f}")
    print(f"   Trades exécutés: {trade_count}")
    print(f"   Changements de portfolio détectés: {len(portfolio_changes)}")
    
    print(f"\n[6] Analyse des résultats:")
    
    # Critères de succès
    success = True
    
    if final_portfolio == initial_portfolio:
        print("   ❌ ÉCHEC: Portfolio n'a pas changé (figé à ${:.2f})".format(initial_portfolio))
        success = False
    else:
        print(f"   ✅ Portfolio a varié")
    
    if trade_count == 0:
        print("   ❌ ÉCHEC: Aucun trade exécuté")
        success = False
    else:
        print(f"   ✅ {trade_count} trades exécutés")
    
    if len(portfolio_changes) == 0:
        print("   ❌ ÉCHEC: Aucun changement de portfolio détecté")
        success = False
    else:
        print(f"   ✅ {len(portfolio_changes)} changements de portfolio détectés")
    
    if len(portfolio_changes) > 0:
        print(f"\n[7] Détails des changements de portfolio:")
        for i, change in enumerate(portfolio_changes[:10], 1):  # Max 10 premiers
            print(f"   {i}. Step {change['step']:3d}: "
                  f"${change['old']:.2f} → ${change['new']:.2f} "
                  f"(Δ=${change['delta']:.4f})")
    
    print("\n" + "=" * 80)
    if success:
        print("VERDICT: ✅ TEST 1.1 RÉUSSI")
        print("L'environnement fonctionne: trades s'exécutent et portfolio change")
    else:
        print("VERDICT: ❌ TEST 1.1 ÉCHOUÉ")
        print("Problème détecté dans l'environnement de base")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = test_env_basic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
