#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic Backtester pour ADAN avec intégration du Dynamic Behavior Engine (DBE).

Ce script effectue un backtest en simulant le comportement en temps réel,
y compris le chargement par chunks et l'adaptation dynamique des paramètres
via le DBE.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import torch
import logging

from stable_baselines3 import PPO
from src.adan_trading_bot.common.utils import load_config
from src.adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv,
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dynamic_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DynamicBacktester')

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))


# Configuration des chemins
REPORTS_DIR = Path('reports/backtests')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class DynamicBacktester:
    """
    Classe pour effectuer des backtests dynamiques avec intégration du DBE.
    """

    def __init__(self, config_path: str, model_path: str):
        """
        Initialise le backtester dynamique.

        Args:
            config_path: Chemin vers le fichier de configuration
            model_path: Chemin vers le modèle entraîné
        """
        self.config = load_config(config_path)
        self.model_path = Path(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialisation des métriques
        self.metrics = {
            'portfolio_value': [],
            'returns': [],
            'trades': [],
            'dbe_metrics': [],
            'timestamp': [],
            'actions': []
        }

        # Initialisation de l'environnement
        self.env = MultiAssetChunkedEnv(self.config)

        logger.info("DynamicBacktester initialisé avec succès")

    def _update_metrics(self, info: Dict[str, Any], action: np.ndarray):
        """Met à jour les métriques avec les informations du pas actuel."""
        self.metrics['portfolio_value'].append(
            info.get('portfolio_value', 0)
        )
        self.metrics['returns'].append(info.get('return', 0))
        self.metrics['timestamp'].append(
            info.get('timestamp', datetime.now())
        )
        self.metrics['actions'].append(
            action.tolist() if action is not None else None
        )

        # Ajouter les métriques du DBE si disponibles
        if hasattr(self.env, 'dbe') and self.env.dbe is not None:
            self.metrics['dbe_metrics'].append({
                'sl_pct': self.env.dbe.state.get('sl_pct', 0),
                'tp_pct': self.env.dbe.state.get('tp_pct', 0),
                'risk_mode': self.env.dbe.state.get('risk_mode', 'NORMAL'),
                'position_size': self.env.dbe.state.get('position_size', 0),
                'market_regime': self.env.dbe.state.get(
                    'market_regime', 'UNKNOWN'
                ),
                'volatility': self.env.dbe.state.get('volatility', 0)
            })

    def _generate_report(self, output_dir: Path):
        """Génère un rapport de backtest complet."""
        logger.info("Génération du rapport de backtest...")

        # Créer le répertoire de sortie s'il n'existe pas
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convertir les métriques en DataFrame pour l'analyse
        df_metrics = pd.DataFrame({
            'timestamp': self.metrics['timestamp'],
            'portfolio_value': self.metrics['portfolio_value'],
            'return': self.metrics['returns'],
            'action': self.metrics['actions']
        })

        # Ajouter les métriques du DBE si disponibles
        if self.metrics['dbe_metrics']:
            df_dbe = pd.DataFrame(self.metrics['dbe_metrics'])
            df_metrics = pd.concat([df_metrics, df_dbe], axis=1)

        # Sauvegarder les données brutes
        raw_data_path = output_dir / 'backtest_metrics.csv'
        df_metrics.to_csv(raw_data_path, index=False)

        # Générer le rapport HTML avec quantstats
        self._generate_quantstats_report(df_metrics, output_dir)

        # Générer des graphiques supplémentaires
        self._generate_plots(df_metrics, output_dir)

        logger.info(f"Rapport de backtest généré dans {output_dir}")

    def _generate_quantstats_report(
        self, df_metrics: pd.DataFrame, output_dir: Path
    ):
        """Génère un rapport quantstats à partir des métriques."""
        returns = pd.Series(
            df_metrics['return'].values,
            index=pd.to_datetime(df_metrics['timestamp'])
        )

        # Générer le rapport HTML
        report_path = output_dir / 'quantstats_report.html'
        qs.reports.html(
            returns,
            output=report_path,
            title='ADAN Dynamic Backtest Report',
            download_filename=report_path.name,
        )

    def _generate_plots(self, df_metrics: pd.DataFrame, output_dir: Path):
        """Génère des graphiques supplémentaires."""
        plt.figure(figsize=(15, 10))

        # Graphique de la valeur du portefeuille
        plt.subplot(2, 1, 1)
        plt.plot(df_metrics['timestamp'], df_metrics['portfolio_value'])
        plt.title('Valeur du portefeuille au fil du temps')
        plt.xlabel('Date')
        plt.ylabel('Valeur (USD)')
        plt.grid(True)

        # Graphique des paramètres du DBE
        if 'sl_pct' in df_metrics.columns:
            plt.subplot(2, 1, 2)
            plt.plot(
                df_metrics['timestamp'],
                df_metrics['sl_pct'],
                label='Stop Loss %',
            )
            plt.plot(
                df_metrics['timestamp'],
                df_metrics['tp_pct'],
                label='Take Profit %',
            )
            plt.title('Évolution des paramètres de risque')
            plt.xlabel('Date')
            plt.ylabel('Valeur (%)')
            plt.legend()
            plt.grid(True)

        # Sauvegarder la figure
        plot_path = output_dir / 'backtest_plots.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def run(self, num_episodes: int = 10, render: bool = False):
        """
        Exécute le backtest dynamique.

        Args:
            num_episodes: Nombre d'épisodes à exécuter
            render: Si True, affiche la progression
        """
        logger.info(
            f"Démarrage du backtest dynamique sur {num_episodes} épisodes"
        )

        # Charger le modèle ici, une seule fois
        logger.info(f"Chargement du modèle depuis {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé à l'emplacement {self.model_path}"
            )

        try:
            model = PPO.load(self.model_path, device=self.device)
            logger.info("Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise

        for episode in range(1, num_episodes + 1):
            logger.info(f"Début de l'épisode {episode}/{num_episodes}")

            # Réinitialiser l'environnement
            obs, info_reset = self.env.reset()
            done = False
            episode_reward = 0

            # Boucle sur les pas de temps
            while not done:
                # Sélectionner une action
                action, _ = model.predict(obs, deterministic=True)

                # Exécuter l'action
                next_obs, reward, terminated, truncated, info_step = (
                    self.env.step(action)
                )

                # Mettre à jour les métriques
                # info_step contient les métriques du portefeuille et autres
                # infos de l'environnement
                self._update_metrics(info_step, action)

                # Mettre à jour l'observation
                obs = next_obs
                episode_reward += reward
                done = terminated or truncated

                # Afficher la progression si demandé
                if render:
                    self.env.render()

            logger.info(
                f"Épisode {episode} terminé - Récompense: {episode_reward:.2f}"
            )

        # Générer le rapport final
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = REPORTS_DIR / f'backtest_{timestamp}'
        self._generate_report(report_dir)

        logger.info(f"Backtest terminé. Rapport généré dans {report_dir}")
        return report_dir


def main():
    """Fonction principale pour exécuter le backtest dynamique."""
    parser = argparse.ArgumentParser(
        description='Exécute un backtest dynamique avec DBE'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/trading_config.yaml',
        help='Chemin vers le fichier de configuration',
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Chemin vers le modèle entraîné',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help="Nombre d'épisodes à exécuter",
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Afficher la progression du backtest',
    )

    args = parser.parse_args()

    try:
        # Initialiser le backtester
        backtester = DynamicBacktester(
            config_path=args.config, model_path=args.model
        )

        # Exécuter le backtest
        report_dir = backtester.run(
            num_episodes=args.episodes, render=args.render
        )

        # Ouvrir le rapport dans le navigateur
        report_path = report_dir / 'quantstats_report.html'
        if report_path.exists():
            url = f'file://{report_path.absolute()}'
            webbrowser.open(url)

        return 0
    except Exception as e:
        logger.error(
            f"Erreur lors de l'exécution du backtest: {e}", exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
