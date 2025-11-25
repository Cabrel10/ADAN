#!/usr/bin/env python3
"""
ÉVALUATION COMPLÈTE DU MODÈLE - BTC ET XRP SUR LES PLUS GROS PARQUETS
Date: 2025-11-25
Objectif: Tester le modèle sur les données maximales disponibles
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv
)
from adan_trading_bot.common.config_loader import ConfigLoader


class ComprehensiveEvaluator:
    """Évaluation complète du modèle"""

    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.env = None
        self.results = {}

        logger.info("=" * 80)
        logger.info("ÉVALUATION COMPLÈTE DU MODÈLE")
        logger.info("=" * 80)

    def load_model(self):
        """Charge le modèle"""
        logger.info(f"\n📦 Chargement modèle: {self.model_path}")
        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)
            config['initial_capital'] = 20.5

            self.env = MultiAssetChunkedEnv(
                config=config,
                worker_id=0,
                log_level="WARNING",
                data_split="train"
            )

            self.model = PPO.load(self.model_path, env=self.env)
            logger.info("✅ Modèle chargé")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            return False

    def evaluate_dataset(self, asset, split, max_steps=5000):
        """Évalue sur un dataset"""
        logger.info(f"\n{'='*80}")
        logger.info(f"ÉVALUATION: {asset} ({split})")
        logger.info(f"{'='*80}")

        try:
            # Recréer l'env avec le bon split
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)
            config['initial_capital'] = 20.5
            config['environment']['assets'] = [asset]

            env = MultiAssetChunkedEnv(
                config=config,
                worker_id=0,
                log_level="WARNING",
                data_split=split
            )

            obs, info = env.reset()
            portfolio_values = []
            trades = 0
            steps = 0

            logger.info(f"Démarrage évaluation ({max_steps} steps max)...")

            for step in range(max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                # Récupérer la valeur du portfolio depuis info
                if 'portfolio_value' in info:
                    portfolio_values.append(info['portfolio_value'])
                elif hasattr(env, 'portfolio_value'):
                    portfolio_values.append(env.portfolio_value)

                # Compter les trades
                if 'trade_executed' in info and info['trade_executed']:
                    trades += 1

                steps += 1

                if (step + 1) % 1000 == 0:
                    if portfolio_values:
                        current_value = portfolio_values[-1]
                        return_pct = ((current_value - 20.5) / 20.5) * 100
                        logger.info(
                            f"  Step {step + 1}: Portfolio=${current_value:.2f} "
                            f"({return_pct:+.2f}%), Trades={trades}"
                        )

                if terminated or truncated:
                    logger.info(f"  Épisode terminé au step {step + 1}")
                    break

            # Calculer les métriques
            if not portfolio_values:
                logger.warning("Aucune valeur de portfolio collectée!")
                portfolio_values = [20.5]

            portfolio_array = np.array(portfolio_values)
            final_value = portfolio_array[-1]
            max_value = np.max(portfolio_array)
            min_value = np.min(portfolio_array)

            total_return = ((final_value - 20.5) / 20.5) * 100
            max_drawdown = ((min_value - max_value) / max_value) * 100

            result = {
                'asset': asset,
                'split': split,
                'steps': steps,
                'trades': trades,
                'initial_capital': 20.5,
                'final_capital': final_value,
                'total_return_pct': total_return,
                'max_drawdown_pct': max_drawdown,
                'max_portfolio': max_value,
                'min_portfolio': min_value,
            }

            logger.info(f"\n📊 RÉSULTATS: {asset} ({split})")
            logger.info(f"  Steps: {steps}")
            logger.info(f"  Trades: {trades}")
            logger.info(f"  Capital: ${20.5:.2f} → ${final_value:.2f}")
            logger.info(f"  Return: {total_return:+.2f}%")
            logger.info(f"  Max Drawdown: {max_drawdown:.2f}%")

            return result

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'évaluation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_comprehensive_evaluation(self):
        """Lance l'évaluation complète"""
        if not self.load_model():
            return

        # Évaluer sur les plus gros parquets
        datasets = [
            ('XRPUSDT', 'train', 5000),  # 30M - le plus gros
            ('BTCUSDT', 'test', 5000),   # 8.6M
            ('XRPUSDT', 'test', 5000),   # 7.6M
            ('BTCUSDT', 'train', 5000),  # 3.3M
        ]

        for asset, split, max_steps in datasets:
            result = self.evaluate_dataset(asset, split, max_steps)
            if result:
                key = f"{asset}_{split}"
                self.results[key] = result

        # Résumé final
        self.print_summary()

    def print_summary(self):
        """Affiche un résumé"""
        logger.info(f"\n{'='*80}")
        logger.info("RÉSUMÉ FINAL")
        logger.info(f"{'='*80}\n")

        if not self.results:
            logger.warning("Aucun résultat")
            return

        # Tableau
        logger.info("Dataset              | Return  | DD      | Trades | Steps")
        logger.info("-" * 65)

        total_return = 0
        total_trades = 0
        total_steps = 0

        for key, result in self.results.items():
            logger.info(
                f"{key:20} | "
                f"{result['total_return_pct']:+7.2f}% | "
                f"{result['max_drawdown_pct']:7.2f}% | "
                f"{result['trades']:6} | "
                f"{result['steps']:5}"
            )
            total_return += result['total_return_pct']
            total_trades += result['trades']
            total_steps += result['steps']

        logger.info("-" * 65)
        avg_return = total_return / len(self.results)
        logger.info(
            f"{'MOYENNE':20} | "
            f"{avg_return:+7.2f}% | "
            f"{'':7} | "
            f"{total_trades:6} | "
            f"{total_steps:5}"
        )

        logger.info(f"\n{'='*80}")
        logger.info("✅ ÉVALUATION COMPLÈTE TERMINÉE")
        logger.info(f"{'='*80}")

        # Verdict
        if avg_return > 100:
            logger.info("🎯 VERDICT: ✅ EXCELLENT (>100% return)")
        elif avg_return > 50:
            logger.info("🎯 VERDICT: ✅ BON (>50% return)")
        elif avg_return > 0:
            logger.info("🎯 VERDICT: ✅ POSITIF (>0% return)")
        else:
            logger.info("🎯 VERDICT: ❌ NÉGATIF (<0% return)")


def main():
    evaluator = ComprehensiveEvaluator(
        model_path="checkpoints_final/adan_model_checkpoint_640000_steps.zip",
        config_path="config/config.yaml"
    )
    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
