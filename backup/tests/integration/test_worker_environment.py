#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test d'intégration pour vérifier que MultiAssetChunkedEnv s'initialise correctement
avec des configurations de worker spécifiques.
"""

import os, sys
# On se place toujours au root du projet
ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import unittest
from typing import Dict, Any

import pandas as pd
import yaml

print("--- DEBUG: Imports de base terminés ---")

# L'import doit être fait après avoir configuré le PYTHONPATH
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

print("--- DEBUG: PYTHONPATH modifié, sur le point d'importer MultiAssetChunkedEnv ---")
print("--- DEBUG: Importation de MultiAssetChunkedEnv réussie ---")


def load_full_config() -> dict:
    """Charge la configuration complète depuis config.yaml."""
    config_path = os.path.join(ROOT, "../config/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


print("--- DEBUG: Définition de la classe de test sur le point de commencer ---")


class TestWorkerEnvironment(unittest.TestCase):
    """Valide l'initialisation de l'environnement par worker."""

    @classmethod
    def setUpClass(cls):
        """Charger la configuration une seule fois pour tous les tests."""
        cls.full_config = load_full_config()
        if 'workers' not in cls.full_config or not cls.full_config['workers']:
            raise unittest.SkipTest("Section 'workers' non trouvée ou vide dans config.yaml")

    def test_worker_1_temporal_precision_env(self):
        """Vérifie la configuration de l'environnement pour le Worker 1."""
        worker_config = self.full_config['workers'].get('w1')
        self.assertIsNotNone(worker_config, "Worker 'w1' non trouvé dans la config")

        # Initialiser l'environnement pour ce worker
        env = MultiAssetChunkedEnv(
            config=self.full_config,
            worker_config=worker_config
        )

        # 1. Vérifier les timeframes chargés
        self.assertEqual(
            env.timeframes,
            ['5m'],
            "Le Worker 1 doit charger uniquement le timeframe '5m'"
        )

        # 2. Vérifier que le reward calculator est correctement initialisé
        self.assertIsNotNone(
            env.reward_calculator,
            "Le RewardCalculator doit être initialisé pour le Worker 1"
        )

    def test_worker_2_low_frequency_sentinel_env(self):
        """Vérifie la configuration de l'environnement pour le Worker 2."""
        worker_config = self.full_config['workers'].get('w2')
        self.assertIsNotNone(worker_config, "Worker 'w2' non trouvé dans la config")

        # Initialiser l'environnement
        env = MultiAssetChunkedEnv(
            config=self.full_config,
            worker_config=worker_config
        )

        # 1. Vérifier les timeframes
        expected_timeframes = ['1h', '4h']
        self.assertCountEqual(
            env.timeframes,
            expected_timeframes,
            "Le Worker 2 doit charger les timeframes '1h' et '4h'"
        )

    def test_all_workers_initialization(self):
        """Vérifie que tous les workers peuvent initialiser un environnement sans erreur."""
        for worker_id, worker_config in self.full_config['workers'].items():
            with self.subTest(worker=worker_id):
                try:
                    env = MultiAssetChunkedEnv(
                        config=self.full_config,
                        worker_config=worker_config
                    )
                    # Vérification simple
                    self.assertIsNotNone(
                        env.observation_space,
                        f"L'espace d'observation est nul pour {worker_id}"
                    )
                    self.assertIsNotNone(
                        env.action_space,
                        f"L'espace d'action est nul pour {worker_id}"
                    )
                except Exception as e:
                    self.fail(
                        f"L'initialisation de l'environnement a échoué pour {worker_id} "
                        f"avec l'erreur: {e}"
                    )


if __name__ == '__main__':
    print("--- DEBUG: Entrée dans le bloc __main__, lancement de unittest.main() ---")
    unittest.main()
