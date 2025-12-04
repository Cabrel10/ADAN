#!/usr/bin/env python3
"""
TEST 1.2: Vérifier le système de Force Trade
Objectif: Confirmer que les force trades fonctionnent et que min_order_value est correct
"""
import sys
import yaml
import numpy as np
from pathlib import Path

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

def test_force_trade_system():
    print("=" * 80)
    print("TEST 1.2: SYSTÈME DE FORCE TRADE")
    print("=" * 80)
    
    config_path = "config/config.yaml"
    
    print(f"\n[1] Chargement et modification de la configuration")
    with open(config_path,  'r') as f:
        config = yaml.safe_load(f)
    
    # DÉSACTIVER les trades naturels pour tester uniquement force trades
    if 'environment' not in config:
        config['environment'] = {}
    
    # Mettre des seuils d'action très hauts pour empêcher trades naturels
    config['environment']['action_thresholds'] = {
        '5m': 0.99,  # Quasi impossible d'atteindre
        '1h': 0.99,
        '4h': 0.99
    }
    
    # Configurer force trade agressif
    if 'trading_rules' not in config:
        config['trading_rules'] = {}
    if 'frequency' not in config['trading_rules']:
        config['trading_rules']['frequency'] = {}
    
    config['trading_rules']['frequency']['force_trade_steps'] = {
        '5m': 10,  # Force trade tous les 10 steps
        '1h': 999,  # Désactiver 1h
        '4h': 999   # Désactiver 4h
    }
    
    # Vérifier min_order_value
    if 'dbe' in config:
        min_value = config['dbe'].get('risk_management', {}).get('min_trade_value', 'NOT SET')
        print(f"   min_trade_value actuel: {min_value}")
        if min_value == 'NOT SET' or float(min_value) > 10:
            print(f"   ⚠️  ATTENTION: min_trade_value = {min_value} peut bloquer avec capital $20.50")
            config['dbe'].setdefault('risk_management', {})['min_trade_value'] = 1.0
            print(f"   ✅ Ajusté à 1.0 USDT")
    
    print(f"✅ Config modifiée: action_threshold=0.99, force_trade_5m=10 steps")
    
    print(f"\n[2] Initialisation de l'environnement")
    env = MultiAssetChunkedEnv(config=config)
    
    print(f"\n[3] Reset")
    obs = env.reset()
    initial_portfolio = env.portfolio.portfolio_value
    
    print(f"   Portfolio initial: ${initial_portfolio:.2f}")
    
    print(f"\n[4] Exécution de 50 steps avec actions NULLES (devrait forcer trades)")
    force_trades_count = 0
    natural_trades_count = 0
    
    for step in range(50):
        # Action complètement nulle
        action = np.zeros(env.action_space.shape)
        
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
        
        # Compter force vs natural trades dans les logs
        # Pas d'accès direct, mais on peut voir le portfolio
        
        if step % 10 == 9:
            current_portfolio = env.portfolio.portfolio_value
            print(f"   Step {step+1:2d}: portfolio=${current_portfolio:.2f}")
        
        if done:
            break
    
    final_portfolio = env.portfolio.portfolio_value
    
    print(f"\n[5] Résultats:")
    print(f"   Portfolio initial: ${initial_portfolio:.2f}")
    print(f"   Portfolio final: ${final_portfolio:.2f}")
    print(f"   Positions actives: {len(env.portfolio.positions)}")
    
    # Vérifier que des trades ont été forcés
    success = True
    
    if final_portfolio == initial_portfolio and len(env.portfolio.positions) == 0:
        print("\n   ❌ ÉCHEC: Aucun trade exécuté malgré force trade activé")
        success = False
    else:
        print(f"\n   ✅ Des trades ont été exécutés (portfolio ou positions ont changé)")
    
    # Vérifier min_order_value
    print(f"\n[6] Vérification min_order_value:")
    if hasattr(env.portfolio, 'min_order_value_usdt'):
        min_val = env.portfolio.min_order_value_usdt
        print(f"   min_order_value_usdt = {min_val}")
        if min_val <= 1.0:
            print(f"   ✅ min_order_value ≤ $1.0 (compatible avec capital $20.50)")
        else:
            print(f"   ⚠️  min_order_value = ${min_val} peut bloquer petit capital")
            if min_val >= 10:
                print(f"   ❌ PROBLÈME: ${min_val} est trop élevé pour $20.50")
                success = False
    else:
        print(f"   ⚠️  min_order_value_usdt attribut non trouvé")
    
    print("\n" + "=" * 80)
    if success:
        print("VERDICT: ✅ TEST 1.2 RÉUSSI")
        print("Force trades fonctionnent, min_order_value correct")
    else:
        print("VERDICT: ❌ TEST 1.2 ÉCHOUÉ")
        print("Force trades ou min_order_value défectueux")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = test_force_trade_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
