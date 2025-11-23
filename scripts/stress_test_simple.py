#!/usr/bin/env python3
"""
STRESS TEST SIMPLE - Utilise les données existantes
Teste le modèle sur les périodes critiques disponibles
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv
)
from adan_trading_bot.common.config_loader import ConfigLoader


def check_available_data():
    """Vérifie quelles données sont disponibles"""
    logger.info("Vérification des données disponibles...")

    data_dirs = [
        "data/processed/indicators/train",
        "data/processed/indicators/val",
        "data/processed/indicators/test",
    ]

    available = {}
    for data_dir in data_dirs:
        path = Path(data_dir)
        if path.exists():
            assets = [d.name for d in path.iterdir() if d.is_dir()]
            available[data_dir] = assets
            logger.info(f"  {data_dir}: {assets}")

    return available


def run_backtest_on_split(split: str, asset: str):
    """Lance un backtest sur un split de données"""
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST: {split.upper()} - {asset}")
    logger.info(f"{'='*80}")

    # Charger config
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    config['initial_capital'] = 20.5
    config['environment']['assets'] = [asset]

    # Créer environnement
    try:
        logger.info(f"Création environnement...")
        env = MultiAssetChunkedEnv(
            config=config,
            worker_id=0,
            log_level="WARNING",
            data_split=split
        )
        logger.info("✅ Environnement créé")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return None

    # Charger modèle
    try:
        logger.info("Chargement modèle...")
        model = PPO.load(
            "checkpoints_final/adan_model_checkpoint_640000_steps.zip",
            env=env
        )
        logger.info("✅ Modèle chargé")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return None

    # Backtest
    logger.info("Lancement backtest...")
    obs, _ = env.reset()
    done = False
    step = 0
    portfolio_values = []

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
        return None

    logger.info(f"✅ Backtest terminé: {step} steps")

    # Calculer métriques
    if not portfolio_values:
        logger.error("❌ Pas de données")
        return None

    import numpy as np

    initial = 20.5
    final = portfolio_values[-1]
    ret = (final - initial) / initial * 100

    equity_arr = np.array(portfolio_values)
    peak = np.maximum.accumulate(equity_arr)
    dd = np.min((equity_arr - peak) / peak) * 100

    logger.info(f"\n{'='*80}")
    logger.info(f"RÉSULTATS: {split.upper()} - {asset}")
    logger.info(f"{'='*80}")
    logger.info(f"Capital Initial:        ${initial:.2f}")
    logger.info(f"Capital Final:          ${final:.2f}")
    logger.info(f"Total Return:           {ret:.2f}%")
    logger.info(f"Max Drawdown:           {dd:.2f}%")
    logger.info(f"Total Steps:            {step}")

    if ret > 0:
        logger.info(f"✅ MODÈLE PROFITE")
    else:
        logger.info(f"❌ MODÈLE PERD")

    return {
        "split": split,
        "asset": asset,
        "initial": initial,
        "final": final,
        "return": ret,
        "drawdown": dd,
        "steps": step,
    }


def main():
    logger.info("=" * 80)
    logger.info("STRESS TEST - DONNÉES EXISTANTES")
    logger.info("=" * 80)

    # Vérifier données
    available = check_available_data()

    if not available:
        logger.error("❌ Aucune donnée trouvée")
        return

    # Tester sur chaque split
    results = []

    for split in ["train", "val", "test"]:
        path = f"data/processed/indicators/{split}"
        if path not in available:
            logger.warning(f"⚠️ Split {split} non disponible")
            continue

        assets = available[path]
        for asset in assets:
            result = run_backtest_on_split(split, asset)
            if result:
                results.append(result)

    # Résumé
    logger.info(f"\n{'='*80}")
    logger.info("RÉSUMÉ FINAL")
    logger.info(f"{'='*80}")

    for r in results:
        icon = "✅" if r["return"] > 0 else "❌"
        logger.info(
            f"{icon} {r['split']:5} {r['asset']:10} "
            f"Return={r['return']:7.2f}% DD={r['drawdown']:7.2f}%"
        )

    passed = sum(1 for r in results if r["return"] > 0)
    total = len(results)

    logger.info(f"\n{passed}/{total} tests réussis")

    if passed == total:
        logger.info("✅ MODÈLE PASSE TOUS LES TESTS")
    elif passed >= total * 0.6:
        logger.info("⚠️ MODÈLE PASSE LA PLUPART")
    else:
        logger.info("❌ MODÈLE ÉCHOUE")


if __name__ == "__main__":
    main()
