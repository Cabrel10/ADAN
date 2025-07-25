#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test d'intégration du Dynamic Behavior Engine avec ReplayLogger.

Ce script simule une session de trading et vérifie que les décisions du DBE
sont correctement enregistrées dans les logs.
"""
import os
import sys
import time
import json
import pandas as pd
import sys
import os

# Add the project's 'src' directory parent to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas_ta as ta

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
from src.adan_trading_bot.environment.finance_manager import FinanceManager

from src.adan_trading_bot.common.utils import get_logger

# Configuration
TEST_SAVE_DIR = "test_artifacts/dbe_test"
os.makedirs(TEST_SAVE_DIR, exist_ok=True)

# Initialisation du logger
logger = get_logger()

# Configuration de test
TEST_CONFIG = {
    "dbe": {
        "risk_management": {
            "risk_level_smoothing_factor": 0.6,
            "win_rate_threshold": 0.5,
            "drawdown_threshold": 0.15,
            "consecutive_losses_threshold": 4
        },
        "market_regime": {
            "ema_short_period": 10,
            "ema_long_period": 30,
            "rsi_period": 14,
            "adx_period": 14,
            "atr_period": 14
        },
        "position_sizing": {
            "base_size_pct": 0.05,
            "max_size_pct": 0.20,
            "min_size_pct": 0.01
        },
        "learning_modulation": {
            "learning_rate_factor": 0.5,
            "reward_shaping_factor": 0.5
        }
    }
}

def simulate_market_data(days: int = 60, config: dict = None) -> pd.DataFrame:
    """Génère des données de marché simulées avec des indicateurs techniques."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    base_price = 100.0
    
    # Générer des prix avec différentes phases (tendance, range)
    prices = [base_price]
    for i in range(1, len(dates)):
        if i < len(dates) * 0.4:  # Tendance haussière
            move = np.random.normal(0.0005, 0.01)
        elif i < len(dates) * 0.7:  # Range
            move = np.random.normal(0.0, 0.008)
        else:  # Tendance baissière
            move = np.random.normal(-0.0004, 0.012)
        prices.append(prices[-1] * (1 + move))
    prices = np.array(prices)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
        'close': prices,
        'volume': np.random.lognormal(5, 1, len(dates))
    })

    # Ajouter les indicateurs techniques
    if config:
        regime_cfg = config['dbe']['market_regime']
        df.ta.ema(length=regime_cfg['ema_short_period'], append=True, col_names=('ema_short',))
        df.ta.ema(length=regime_cfg['ema_long_period'], append=True, col_names=('ema_long',))
        df.ta.rsi(length=regime_cfg['rsi_period'], append=True, col_names=('rsi',))
        df.ta.adx(length=regime_cfg['adx_period'], append=True, col_names=('adx',))
        df.ta.atr(length=regime_cfg['atr_period'], append=True, col_names=('atr',))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def run_dbe_simulation():
    """Exécute une simulation complète pour tester le DynamicBehaviorEngine."""
    logger.info("Démarrage de la simulation du DBE...")

    # 1. Initialisation
    fm = FinanceManager(initial_capital=10000, fee_pct=0.001, min_order_usdt=10)
    dbe = DynamicBehaviorEngine(config=TEST_CONFIG['dbe'])
    market_data = simulate_market_data(days=90, config=TEST_CONFIG)

    open_trade = None
    symbol = "BTC/USDT"

    # 2. Boucle de simulation
    for i, row in market_data.iterrows():
        # Mettre à jour le FinanceManager avec le prix actuel
        fm.update_market_value({symbol: row['close']})

        # Si un trade est ouvert, vérifier s'il faut le fermer
        if open_trade:
            if fm.check_close_conditions(symbol, row['high'], row['low']):
                closed_trade = fm.close_trade(symbol, row['close'])
                dbe.on_trade_closed(closed_trade)
                logger.info(f"Trade fermé @ {row['close']:.2f} | PnL: {closed_trade['pnl_pct']:.2f}%")
                open_trade = None

        # Mettre à jour l'état du DBE
        metrics = fm.get_performance_metrics()
        metrics['market_data'] = row
        dbe.update_state(metrics)

        # Obtenir les paramètres dynamiques
        params = dbe.compute_dynamic_modulation()

        # Décision de trading (simplifiée)
        # On ouvre un trade toutes les 48h pour voir l'adaptation
        if not open_trade and i % 48 == 0:
            fm.set_leverage(symbol, params['leverage'])
            open_trade = fm.open_trade(
                symbol=symbol,
                trade_type='long' if dbe.market_regime in ['bull', 'volatile'] else 'short',
                amount_pct=params['position_size_pct'],
                entry_price=row['close'],
                sl_pct=params['sl_pct'],
                tp_pct=params['tp_pct']
            )
            if open_trade:
                logger.info(f"Trade ouvert @ {row['close']:.2f} | SL: {params['sl_pct']:.2%} | TP: {params['tp_pct']:.2%}")

        if i % (24 * 7) == 0: # Log hebdomadaire
            logger.info(dbe.get_status())

    # 3. Test de sauvegarde et de chargement (simplifié)
    logger.info("Test de la sauvegarde de l'état...")
    state_path = os.path.join(TEST_SAVE_DIR, "dbe_state.pkl")
    dbe.save_state(state_path)
    logger.info("Sauvegarde de l'état validée.")

    # 4. Afficher les résultats finaux
    logger.info("Simulation terminée. Résultats finaux:")
    logger.info(dbe.get_status())
    final_metrics = fm.get_performance_metrics()
    logger.info(f"Performance finale du portefeuille: {json.dumps(final_metrics, indent=2)}")

    return True

if __name__ == "__main__":
    import shutil

    # Nettoyer le répertoire de sauvegarde
    if os.path.exists(TEST_SAVE_DIR):
        shutil.rmtree(TEST_SAVE_DIR)
    os.makedirs(TEST_SAVE_DIR, exist_ok=True)

    try:
        success = run_dbe_simulation()
        if success:
            print(f"\n✅ Simulation du DBE terminée avec succès.")
            print(f"Les artefacts de test sont dans: {os.path.abspath(TEST_SAVE_DIR)}")
        else:
            print("\n❌ La simulation du DBE a échoué.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Une erreur est survenue durant la simulation: {e}", exc_info=True)
        print(f"\n❌ La simulation du DBE a rencontré une erreur critique.")
        sys.exit(1)
