#!/usr/bin/env python3
"""
BACKTEST XRP - Validation du modèle sur asset différent
Teste si le modèle généralise bien (entraîné sur BTC, testé sur XRP)
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


def backtest_xrp():
    """Backtest sur XRP"""
    logger.info("=" * 80)
    logger.info("BACKTEST XRP - VALIDATION GÉNÉRALISATION")
    logger.info("=" * 80)
    
    # Vérifier données XRP
    xrp_5m_path = "data/processed/indicators/train/XRPUSDT/5m.parquet"
    if not os.path.exists(xrp_5m_path):
        logger.error(f"❌ Données XRP non trouvées: {xrp_5m_path}")
        return False
    
    df_xrp = pd.read_parquet(xrp_5m_path)
    logger.info(f"✅ Données XRP chargées: {len(df_xrp)} rows")
    logger.info(f"   Période: {df_xrp.index.min()} → {df_xrp.index.max()}")
    
    # Configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    config['initial_capital'] = 20.5
    config['environment']['assets'] = ['XRPUSDT']
    
    logger.info("\nCréation environnement XRP...")
    try:
        env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="WARNING")
        logger.info("✅ Environnement créé")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False
    
    logger.info("Chargement modèle (640k steps)...")
    try:
        model = PPO.load("checkpoints_final/adan_model_checkpoint_640000_steps.zip", env=env)
        logger.info("✅ Modèle chargé")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False
    
    # Backtest
    logger.info("\nLancement backtest XRP...")
    obs, _ = env.reset()
    done = False
    step = 0
    portfolio_values = []
    trades = []
    
    try:
        while not done and step < 100000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            pm = env.portfolio_manager
            portfolio_values.append(pm.equity)
            
            if step % 5000 == 0:
                logger.info(f"  Step {step}: Equity=${pm.equity:.2f}")
            
            step += 1
    
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False
    
    logger.info(f"✅ Backtest terminé: {step} steps")
    
    # Extraire trades
    pm = env.portfolio_manager
    if hasattr(pm, 'trade_log'):
        trade_log = list(pm.trade_log)
        for event in trade_log:
            if event.get('asset') == 'XRPUSDT':
                event_type = (event.get('event') or event.get('action', '')).lower()
                if event_type == 'close':
                    trades.append({
                        'pnl': event.get('pnl', 0),
                        'pnl_pct': event.get('pnl_pct', 0),
                    })
    
    # Calculer métriques
    logger.info("\n" + "=" * 80)
    logger.info("RÉSULTATS XRP")
    logger.info("=" * 80)
    
    initial_capital = 20.5
    final_equity = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0
    
    logger.info(f"\nCapital Initial:        ${initial_capital:.2f}")
    logger.info(f"Capital Final:          ${final_equity:.2f}")
    logger.info(f"Total Return:           {total_return:.2f}%")
    
    # Drawdown
    if portfolio_values:
        equity_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_dd = np.min(drawdown) * 100
        logger.info(f"Max Drawdown:           {max_dd:.2f}%")
    
    # Trades
    logger.info(f"\nTotal Trades (CLOSE):   {len(trades)}")
    if trades:
        pnls = [t['pnl'] for t in trades]
        winning = len([p for p in pnls if p > 0])
        losing = len([p for p in pnls if p < 0])
        win_rate = winning / len(trades) * 100 if len(trades) > 0 else 0
        
        logger.info(f"Winning Trades:         {winning}")
        logger.info(f"Losing Trades:          {losing}")
        logger.info(f"Win Rate:               {win_rate:.2f}%")
        
        if len([p for p in pnls if p < 0]) > 0:
            pf = sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0]))
            logger.info(f"Profit Factor:          {pf:.2f}")
        
        logger.info(f"Total PnL:              ${sum(pnls):.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BACKTEST XRP TERMINÉ")
    logger.info("=" * 80)
    
    # Verdict
    if total_return > 0 and len(trades) > 0:
        logger.info("\n✅ MODÈLE GÉNÉRALISE BIEN SUR XRP")
        return True
    else:
        logger.warning("\n⚠️ Performance faible sur XRP")
        return False


if __name__ == "__main__":
    success = backtest_xrp()
    sys.exit(0 if success else 1)
