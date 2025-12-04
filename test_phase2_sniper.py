#!/usr/bin/env python3
"""
TEST SNIPER - Actions Intentionnelles Fortes
Objectif: Valider que le système TRADE correctement avec signaux forts
et calcule PnL, metrics, etc.
"""
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

def test_sniper():
    print("=" * 80)
    print("TEST SNIPER - VALIDATION PIPELINE COMPLET")
    print("=" * 80)
    
    print(f"\n[1] Chargement config")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Désactiver force trades pour clarté
    config['trading_rules']['frequency']['force_trade_steps'] = {
        '5m': 99999, '1h': 99999, '4h': 99999
    }
    
    print(f"\n[2] Initialisation environnement")
    env = MultiAssetChunkedEnv(config=config)
    obs = env.reset()
    
    initial_portfolio = env.portfolio.portfolio_value
    print(f"   Portfolio initial: ${initial_portfolio:.2f}")
    print(f"   Action space shape: {env.action_space.shape}")
    
    print(f"\n[3] Phase SILENCE (steps 0-9): Actions = 0")
    for step in range(10):
        action = np.zeros(env.action_space.shape)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
    
    portfolio_pre_buy = env.portfolio.portfolio_value
    positions_pre_buy = len(env.portfolio.positions)
    print(f"   Après silence: portfolio=${portfolio_pre_buy:.2f}, positions={positions_pre_buy}")
    
    print(f"\n[4] Phase BUY FORT (step 10): Action = 0.3 sur premier actif")
    # Action forte BUY: mettre 0.3 sur premier élément (BTC 5m probablement)
    strong_buy = np.zeros(env.action_space.shape)
    strong_buy[0] = 0.3  # Signal BUY fort
    
    result = env.step(strong_buy)
    if len(result) == 5:
        obs, reward_buy, done, truncated, info_buy = result
    else:
        obs, reward_buy, done, info_buy = result
    
    portfolio_post_buy = env.portfolio.portfolio_value
    positions_post_buy = len(env.portfolio.positions)
    
    print(f"   Après BUY:")
    print(f"      Portfolio: ${portfolio_pre_buy:.2f} → ${portfolio_post_buy:.2f}")
    print(f"      Positions: {positions_pre_buy} → {positions_post_buy}")
    print(f"      Reward: {reward_buy:.4f}")
    if isinstance(info_buy, dict):
        print(f"      Info keys: {list(info_buy.keys())}")
    
    buy_executed = positions_post_buy > positions_pre_buy
    if buy_executed:
        print(f"   ✅ BUY EXÉCUTÉ!")
    else:
        print(f"   ❌ BUY NON EXÉCUTÉ (problème!)")
    
    print(f"\n[5] Phase HOLD (steps 11-19): Laisser évoluer")
    for step in range(9):
        action = np.zeros(env.action_space.shape)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
        
        if step % 3 == 0:
            current_pf = env.portfolio.portfolio_value
            print(f"   Step {11+step}: portfolio=${current_pf:.2f}")
    
    portfolio_pre_sell = env.portfolio.portfolio_value
    
    print(f"\n[6] Phase SELL FORT (step 20): Action = -0.3")
    strong_sell = np.zeros(env.action_space.shape)
    strong_sell[0] = -0.3  # Signal SELL fort
    
    result = env.step(strong_sell)
    if len(result) == 5:
        obs, reward_sell, done, truncated, info_sell = result
    else:
        obs, reward_sell, done, info_sell = result
    
    portfolio_post_sell = env.portfolio.portfolio_value
    positions_post_sell = len(env.portfolio.positions)
    
    print(f"   Après SELL:")
    print(f"      Portfolio: ${portfolio_pre_sell:.2f} → ${portfolio_post_sell:.2f}")
    print(f"      Positions: {positions_post_buy} → {positions_post_sell}")
    print(f"      Reward: {reward_sell:.4f}")
    
    sell_executed = positions_post_sell < positions_post_buy
    if sell_executed:
        print(f"   ✅ SELL EXÉCUTÉ!")
    else:
        print(f"   ❌ SELL NON EXÉCUTÉ")
    
    print(f"\n[7] ANALYSE RÉSULTATS:")
    
    total_pnl = portfolio_post_sell - initial_portfolio
    
    print(f"   Portfolio: ${initial_portfolio:.2f} → ${portfolio_post_sell:.2f}")
    print(f"   PnL Total: ${total_pnl:.2f} ({(total_pnl/initial_portfolio)*100:.2f}%)")
    print(f"   Trades exécutés: BUY={buy_executed}, SELL={sell_executed}")
    
    # Vérifier métriques si disponibles
    if hasattr(env, 'get_metrics'):
        print(f"\n[8] MÉTRIQUES ENVIRONNEMENT:")
        try:
            metrics = env.get_metrics()
            for k, v in metrics.items():
                print(f"      {k}: {v}")
        except Exception as e:
            print(f"      Erreur: {e}")
    
    print(f"\n[9] VERDICT:")
    
    success = True
    
    if not buy_executed:
        print("   ❌ ÉCHEC: BUY non exécuté malgré signal fort (0.3)")
        success = False
    
    if buy_executed and not sell_executed:
        print("   ⚠️  PARTIEL: BUY OK mais SELL non exécuté")
        success = False
    
    if portfolio_post_sell == initial_portfolio and buy_executed:
        print("   ⚠️  Portfolio identique malgré trade (fees pas appliqués?)")
    
    if buy_executed and sell_executed:
        print("   ✅ PIPELINE COMPLET FONCTIONNE:")
        print("      - Trades exécutés sur signaux forts")
        print("      - Portfolio mis à jour")
        print("      - PnL calculé")
    
    print("\n" + "=" * 80)
    if success:
        print("CONCLUSION: ✅ MOTEUR DE TRADING OPÉRATIONNEL")
        print("\n💡 PROCHAINE ÉTAPE:")
        print("- Test avec actions gaussiennes (PPO-like)")
        print("- Puis training PPO court (1000 steps)")
        print("- Puis Optuna W2/W3/W4")
    else:
        print("CONCLUSION: ❌ PROBLÈME DANS PIPELINE TRADING")
        print("Besoin debug approfondi")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = test_sniper()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
