#!/usr/bin/env python3
"""
STRESS TEST BACKTEST
Lance les backtests sur les 5 scénarios mortels
Teste si le modèle est un suiveur de tendance ou un vrai pro
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv
)
from adan_trading_bot.common.config_loader import ConfigLoader


STRESS_SCENARIOS = {
    "BEAR_2018": {
        "asset": "BTCUSDT",
        "description": "Bear Market 2018 (BTC -85%)",
    },
    "COVID_CRASH": {
        "asset": "BTCUSDT",
        "description": "Crash COVID Mars 2020 (BTC -60% en 2 jours)",
    },
    "ALT_MASSACRE": {
        "asset": "XRPUSDT",
        "description": "Altcoin Massacre 2022 (XRP -80%)",
    },
    "DEAD_RANGE": {
        "asset": "BTCUSDT",
        "description": "Dead Range 6 mois (0 volatilité, piège)",
    },
}


def run_stress_test(scenario_name: str, scenario_config: dict):
    """
    Lance un backtest sur un scénario de stress
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STRESS TEST: {scenario_name}")
    logger.info(f"Description: {scenario_config['description']}")
    logger.info(f"{'='*80}")

    asset = scenario_config["asset"]

    # Charger config
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    config['initial_capital'] = 20.5
    config['environment']['assets'] = [asset]

    # Créer environnement
    try:
        logger.info(f"Création environnement pour {asset}...")
        env = MultiAssetChunkedEnv(
            config=config,
            worker_id=0,
            log_level="WARNING",
            data_split="stress_tests",
            scenario_name=scenario_name
        )
        logger.info("✅ Environnement créé")
    except Exception as e:
        logger.error(f"❌ Erreur création env: {e}")
        return None

    # Charger modèle
    try:
        logger.info("Chargement modèle (640k steps)...")
        model = PPO.load(
            "checkpoints_final/adan_model_checkpoint_640000_steps.zip",
            env=env
        )
        logger.info("✅ Modèle chargé")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
        return None

    # Backtest
    logger.info("Lancement backtest...")
    obs, _ = env.reset()
    done = False
    step = 0
    portfolio_values = []
    trades_count = 0

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
        logger.error(f"❌ Erreur backtest: {e}")
        return None

    logger.info(f"✅ Backtest terminé: {step} steps")

    # Calculer métriques
    if not portfolio_values:
        logger.error("❌ Pas de données de portfolio")
        return None

    initial_capital = 20.5
    final_equity = portfolio_values[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100

    # Drawdown
    equity_array = np.array(portfolio_values)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - peak) / peak
    max_dd = np.min(drawdown) * 100

    # Sharpe ratio
    returns = np.diff(equity_array) / equity_array[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 12)

    results = {
        "scenario": scenario_name,
        "asset": asset,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "steps": step,
        "status": "SUCCESS",
    }

    # Log résultats
    logger.info(f"\n{'='*80}")
    logger.info(f"RÉSULTATS {scenario_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Capital Initial:        ${initial_capital:.2f}")
    logger.info(f"Capital Final:          ${final_equity:.2f}")
    logger.info(f"Total Return:           {total_return:.2f}%")
    logger.info(f"Max Drawdown:           {max_dd:.2f}%")
    logger.info(f"Sharpe Ratio:           {sharpe:.2f}")
    logger.info(f"Total Steps:            {step}")

    # Verdict
    if total_return > 0 and max_dd > -50:
        logger.info(f"\n✅ MODÈLE SURVIT AU STRESS")
    elif total_return > 0:
        logger.info(f"\n⚠️ MODÈLE SURVIT MAIS DRAWDOWN ÉLEVÉ")
    else:
        logger.info(f"\n❌ MODÈLE ÉCHOUE AU STRESS")

    return results


def run_all_stress_tests():
    """
    Lance tous les stress tests
    """
    logger.info("=" * 80)
    logger.info("STRESS TEST SUITE - 5 SCÉNARIOS MORTELS")
    logger.info("=" * 80)

    results = {}

    for scenario_name, scenario_config in STRESS_SCENARIOS.items():
        result = run_stress_test(scenario_name, scenario_config)
        if result:
            results[scenario_name] = result

    # Résumé final
    logger.info(f"\n{'='*80}")
    logger.info("RÉSUMÉ FINAL")
    logger.info(f"{'='*80}")

    for scenario_name, result in results.items():
        status_icon = "✅" if result["total_return"] > 0 else "❌"
        logger.info(
            f"{status_icon} {scenario_name}: "
            f"Return={result['total_return']:.2f}%, "
            f"DD={result['max_drawdown']:.2f}%"
        )

    # Verdict global
    passed = sum(1 for r in results.values() if r["total_return"] > 0)
    total = len(results)

    logger.info(f"\n{'='*80}")
    logger.info(f"VERDICT GLOBAL: {passed}/{total} scénarios réussis")
    logger.info(f"{'='*80}")

    if passed == total:
        logger.info("✅ MODÈLE PASSE TOUS LES STRESS TESTS")
    elif passed >= total * 0.6:
        logger.info("⚠️ MODÈLE PASSE LA PLUPART DES TESTS")
    else:
        logger.info("❌ MODÈLE ÉCHOUE LES STRESS TESTS")

    return results


if __name__ == "__main__":
    try:
        results = run_all_stress_tests()

        # Sauvegarder résultats
        output_dir = Path("stress_tests/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(results).T
        results_df.to_csv(output_dir / "stress_test_results.csv")

        logger.info(f"\n✅ Résultats sauvegardés dans {output_dir}")

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)
