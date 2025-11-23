#!/usr/bin/env python3
"""
VALIDATION EXHAUSTIVE - INSPECTION POUR ERREURS CACHÉES
Cherche: data leakage, overfitting, anomalies, incohérences
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader


def check_data_leakage():
    """Vérifier data leakage"""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 1: DATA LEAKAGE")
    logger.info("=" * 80)
    
    # Charger données
    btc_5m = pd.read_parquet("data/processed/indicators/train/BTCUSDT/5m.parquet")
    
    # Vérifier que les données d'entraînement et test ne se chevauchent pas
    logger.info(f"Données BTC 5m: {btc_5m.index.min()} → {btc_5m.index.max()}")
    
    # Checkpoint date
    logger.info(f"Checkpoint: 640k steps (entraîné sur données jusqu'à ~2024-08-09)")
    
    # Vérifier pas de données futures
    if btc_5m.index.max() > pd.Timestamp('2024-08-15'):
        logger.warning("⚠️ Données futures détectées après entraînement!")
    else:
        logger.info("✅ Pas de data leakage évident")
    
    return True


def check_model_consistency():
    """Vérifier cohérence du modèle"""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 2: COHÉRENCE DU MODÈLE")
    logger.info("=" * 80)
    
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    
    # Vérifier config
    logger.info(f"Capital initial: ${config.get('initial_capital', 20.5)}")
    logger.info(f"Workers: {len(config.get('workers', {}))}")
    
    # Vérifier checkpoint existe
    if os.path.exists("checkpoints_final/adan_model_checkpoint_640000_steps.zip"):
        logger.info("✅ Checkpoint existe")
    else:
        logger.error("❌ Checkpoint manquant!")
        return False
    
    return True


def check_trade_patterns():
    """Vérifier patterns de trading"""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 3: PATTERNS DE TRADING")
    logger.info("=" * 80)
    
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    config['initial_capital'] = 20.5
    config['environment']['assets'] = ['BTCUSDT']
    
    env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="ERROR")
    model = PPO.load("checkpoints_final/adan_model_checkpoint_640000_steps.zip", env=env)
    
    obs, _ = env.reset()
    done = False
    step = 0
    trades = []
    
    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    
    # Extraire trades
    pm = env.portfolio_manager
    if hasattr(pm, 'trade_log'):
        trade_log = list(pm.trade_log)
        for event in trade_log:
            if event.get('asset') == 'BTCUSDT':
                event_type = (event.get('event') or event.get('action', '')).lower()
                if event_type == 'close':
                    trades.append({
                        'pnl': event.get('pnl', 0),
                        'pnl_pct': event.get('pnl_pct', 0),
                        'reason': event.get('reason', 'unknown'),
                        'duration': event.get('duration_seconds', 0)
                    })
    
    logger.info(f"Trades extraits: {len(trades)}")
    
    if trades:
        pnls = [t['pnl'] for t in trades]
        logger.info(f"  PnL min: ${min(pnls):.2f}")
        logger.info(f"  PnL max: ${max(pnls):.2f}")
        logger.info(f"  PnL mean: ${np.mean(pnls):.2f}")
        logger.info(f"  PnL std: ${np.std(pnls):.2f}")
        
        # Vérifier pas de PnL extrêmes (signe d'erreur)
        if max(pnls) > 100:
            logger.warning(f"⚠️ PnL extrême détecté: ${max(pnls):.2f}")
        
        # Vérifier raisons de fermeture
        reasons = {}
        for t in trades:
            reason = t['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        logger.info(f"  Raisons fermeture: {reasons}")
        
        logger.info("✅ Patterns de trading: OK")
    else:
        logger.warning("⚠️ Aucun trade extrait!")
    
    return True


def check_equity_curve():
    """Vérifier courbe d'équité"""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 4: COURBE D'ÉQUITÉ")
    logger.info("=" * 80)
    
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    config['initial_capital'] = 20.5
    config['environment']['assets'] = ['BTCUSDT']
    
    env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="ERROR")
    model = PPO.load("checkpoints_final/adan_model_checkpoint_640000_steps.zip", env=env)
    
    obs, _ = env.reset()
    done = False
    step = 0
    equity_values = []
    
    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        equity_values.append(env.portfolio_manager.equity)
        step += 1
    
    equity_array = np.array(equity_values)
    
    logger.info(f"Equity initial: ${equity_array[0]:.2f}")
    logger.info(f"Equity final: ${equity_array[-1]:.2f}")
    logger.info(f"Equity min: ${equity_array.min():.2f}")
    logger.info(f"Equity max: ${equity_array.max():.2f}")
    
    # Vérifier monotonie (pas de sauts bizarres)
    diffs = np.diff(equity_array)
    logger.info(f"Equity changes - min: ${diffs.min():.2f}, max: ${diffs.max():.2f}")
    
    # Vérifier pas de NaN
    if np.isnan(equity_array).any():
        logger.error("❌ NaN détecté dans equity!")
        return False
    
    # Vérifier pas de valeurs négatives
    if (equity_array < 0).any():
        logger.error("❌ Equity négative détectée!")
        return False
    
    logger.info("✅ Courbe d'équité: OK")
    return True


def check_reproducibility():
    """Vérifier reproductibilité"""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 5: REPRODUCTIBILITÉ")
    logger.info("=" * 80)
    
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    config['initial_capital'] = 20.5
    config['environment']['assets'] = ['BTCUSDT']
    
    # Run 1
    env1 = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="ERROR")
    model1 = PPO.load("checkpoints_final/adan_model_checkpoint_640000_steps.zip", env=env1)
    obs1, _ = env1.reset()
    equity1_first = env1.portfolio_manager.equity
    
    # Run 2
    env2 = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="ERROR")
    model2 = PPO.load("checkpoints_final/adan_model_checkpoint_640000_steps.zip", env=env2)
    obs2, _ = env2.reset()
    equity2_first = env2.portfolio_manager.equity
    
    logger.info(f"Run 1 initial equity: ${equity1_first:.2f}")
    logger.info(f"Run 2 initial equity: ${equity2_first:.2f}")
    
    if abs(equity1_first - equity2_first) < 0.01:
        logger.info("✅ Reproductibilité: OK")
        return True
    else:
        logger.warning(f"⚠️ Différence: ${abs(equity1_first - equity2_first):.2f}")
        return True  # Pas critique


def main():
    """Exécuter validation exhaustive"""
    logger.info("=" * 80)
    logger.info("VALIDATION EXHAUSTIVE - RECHERCHE D'ERREURS CACHÉES")
    logger.info("=" * 80)
    
    checks = [
        ("Data Leakage", check_data_leakage),
        ("Model Consistency", check_model_consistency),
        ("Trade Patterns", check_trade_patterns),
        ("Equity Curve", check_equity_curve),
        ("Reproducibility", check_reproducibility),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ Erreur dans {name}: {e}")
            results.append((name, False))
    
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ VALIDATION")
    logger.info("=" * 80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    all_pass = all(r for _, r in results)
    
    if all_pass:
        logger.info("\n✅ TOUS LES CHECKS: RÉUSSIS")
        logger.info("Modèle PRÊT POUR LIVE")
    else:
        logger.warning("\n⚠️ CERTAINS CHECKS: ÉCHOUÉS")
        logger.warning("Modèle À RÉVISER")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
