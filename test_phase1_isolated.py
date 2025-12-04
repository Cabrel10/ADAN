#!/usr/bin/env python3
"""
TEST ISOLÉ - SIMPLIFICATION RADICALE
Objectif: Forcer config ultra-restrictif EN MÉMOIRE pour isoler source over-trading
"""
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

def test_isolated():
    print("=" * 80)
    print("TEST ISOLÉ - CONFIG FORCÉ EN MÉMOIRE")
    print("=" * 80)
    
    print(f"\n[1] Chargement config de base")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"[2] OVERRIDES EN MÉMOIRE (Simplification radicale)")
    
    # OVERRIDE 1: 1 seul asset
    if 'data' in config:
        config['data']['assets'] = ['BTCUSDT']
        config['data']['include'] = ['BTCUSDT']
    if 'environment' in config:
        config['environment']['assets'] = ['BTCUSDT']
    
    print("   ✅ Assets: 1 seul (BTCUSDT)")
    
    # OVERRIDE 2: 1 seul timeframe
    if 'data' in config:
        config['data']['timeframes'] = ['5m']
    if 'environment' in config:
        if 'observation' in config['environment']:
            config['environment']['observation']['timeframes'] = ['5m']
    if 'trading' in config:
        config['trading']['timeframes'] = ['5m']
    
    print("   ✅ Timeframes: 1 seul (5m)")
    
    # OVERRIDE 3: Action threshold EXTRÊME
    if 'environment' not in config:
        config['environment'] = {}
    config['environment']['action_thresholds'] = {
        '5m': 0.50,  # 50% = Quasi impossible avec actions aléatoires
        '1h': 0.50,
        '4h': 0.50
    }
    
    print("   ✅ Action threshold: 0.50 (EXTRÊME)")
    
    # OVERRIDE 4: Désactiver force trade
    if 'trading_rules' not in config:
        config['trading_rules'] = {}
    if 'frequency' not in config['trading_rules']:
        config['trading_rules']['frequency'] = {}
    
    config['trading_rules']['frequency']['force_trade_steps'] = {
        '5m': 99999,  # Effectivement désactivé
        '1h': 99999,
        '4h': 99999
    }
    
    print("   ✅ Force trade: DÉSACTIVÉ (99999 steps)")
    
    # OVERRIDE 5: Min order value raisonnable
    if 'dbe' in config and 'risk_management' in config['dbe']:
        config['dbe']['risk_management']['min_trade_value'] = 1.0
    
    print("   ✅ Min trade value: 1.0 USDT")
    
    print(f"\n[3] Création environnement avec config forcé")
    env = MultiAssetChunkedEnv(config=config)
    
    print(f"   Env créé: {env.assets}")
    
    print(f"\n[4] Reset")
    obs = env.reset()
    initial_portfolio = env.portfolio.portfolio_value
    
    print(f"   Portfolio initial: ${initial_portfolio:.2f}")
    
    print(f"\n[5] Exécution 50 steps avec actions ALÉATOIRES petites")
    trade_count = 0
    portfolio_history = [initial_portfolio]
    
    for step in range(50):
        # Actions aléatoires PETITES (mean=0, std=0.1)
        # -> Probabilité d'avoir action > 0.5 est TRÈS faible
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
                trade_count += len(trades_info)
            elif isinstance(trades_info, int):
                trade_count += trades_info
            elif trades_info:
                trade_count += 1
        
        if step % 10 == 9:
            print(f"   Step {step+1:2d}: portfolio=${current_portfolio:.2f}, trades_total={trade_count}, max_action={np.max(np.abs(action)):.3f}")
        
        if done:
            break
    
    final_portfolio = env.portfolio.portfolio_value
    portfolio_variance = np.std(portfolio_history)
    
    print(f"\n[6] RÉSULTATS:")
    print(f"   Portfolio: ${initial_portfolio:.2f} → ${final_portfolio:.2f}")
    print(f"   Delta: ${final_portfolio - initial_portfolio:.2f} ({((final_portfolio/initial_portfolio - 1)*100):.2f}%)")
    print(f"   Variance: ${portfolio_variance:.2f}")
    print(f"   Trades TOTAL: {trade_count}")
    print(f"   Trades/step: {trade_count/50:.2f}")
    print(f"   Positions finales: {len(env.portfolio.positions)}")
    
    print(f"\n[7] ANALYSE:")
    
    success = True
    
    if trade_count == 0:
        print("   ✅ EXCELLENT: 0 trades (config ultra-restrictif fonctionne)")
        print("   → Le problème était config trop permissif")
    elif trade_count <= 2:
        print(f"  ✅ BON: {trade_count} trades seulement (chance)")
        print("   → Config restrictif respecté")
    elif trade_count <= 10:
        print(f"   🟡 ACCEPTABLE: {trade_count} trades")
        print("   → Peut-être quelques actions >0.5 par chance")
    else:
        print(f"   ❌ ÉCHEC: {trade_count} trades TROP ÉLEVÉ")
        print("   → BUG LOGIQUE: Config forcé est IGNORÉ")
        success = False
    
    if portfolio_variance < 0.10:
        print(f"   ⚠️  Portfolio très stable (variance={portfolio_variance:.2f})")
        if trade_count > 0:
            print("   → Trades existent mais n'affectent pas portfolio?")
    
    print("\n" + "=" * 80)
    if success:
        print("VERDICT: ✅ TEST ISOLÉ RÉUSSI")
        print("Config restrictif fonctionne - Problème = config trop permissif avant")
        print("\n💡 SOLUTION: Utiliser ces paramètres dans config.yaml principal")
    else:
        print("VERDICT: ❌ TEST ISOLÉ ÉCHOUÉ")
        print("BUG PROFOND: Code ignore action_threshold ou force_trade_steps")
        print("\n🔧 DEBUG REQUIS: Examiner _execute_trades() ligne par ligne")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = test_isolated()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
