#!/usr/bin/env python
"""
Test d'intégration pour le Sprint 3 - Fusion de modèles et surveillance des workers
"""

import os
import sys
import time
import unittest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any

# Ajouter le répertoire src au path Python
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.training.training_orchestrator import TrainingOrchestrator
from adan_trading_bot.monitoring.worker_monitor import WorkerMonitor, TradeRecord

# Configuration de test
TEST_CONFIG = {
    "monitoring_interval": 5,  # secondes
    "sync_frequency": 10,      # pas entre les synchronisations
}

class TestSprint3(unittest.TestCase):
    """Tests d'intégration pour le Sprint 3."""

    def setUp(self):
        """Initialisation des tests."""
        self.orchestrator = TrainingOrchestrator(TEST_CONFIG)
        self.worker_monitor = WorkerMonitor()

        # Créer des modèles factices pour les tests
        self.models = []
        for i in range(3):  # 3 workers
            model = self._create_dummy_model()
            self.models.append(model)
            self.orchestrator.worker_models[f"worker_{i}"] = model

    # Définir la classe DummyPolicy à l'extérieur de _create_dummy_model
    class DummyPolicy:
        def __init__(self):
            # Créer des poids aléatoires
            self.weights = {
                'fc1.weight': torch.randn(64, 10),
                'fc1.bias': torch.randn(64),
                'fc2.weight': torch.randn(32, 64),
                'fc2.bias': torch.randn(32),
                'fc3.weight': torch.randn(2, 32),
                'fc3.bias': torch.randn(2),
            }

        def state_dict(self):
            return self.weights

        def load_state_dict(self, state_dict, strict=True):
            self.weights = state_dict

    # Définir la classe DummyModel une seule fois
    class DummyModel:
        def __init__(self):
            self.policy = TestSprint3.DummyPolicy()

    def _create_dummy_model(self):
        """Crée un modèle factice pour les tests."""
        return TestSprint3.DummyModel()

    def test_model_synchronization(self):
        """Teste la synchronisation des modèles."""
        # Vérifier que les modèles sont initialement différents
        self._assert_models_different()

        # Forcer une synchronisation
        self.orchestrator._synchronize_models(force=True)

        # Vérifier que les modèles sont maintenant identiques
        self._assert_models_identical()

    def _assert_models_different(self):
        """Vérifie que les modèles ont des poids différents."""
        model_states = []
        for model in self.models:
            state = model.policy.state_dict()
            model_states.append(state)

        # Vérifier qu'au moins un paramètre est différent entre les modèles
        for i in range(1, len(model_states)):
            for key in model_states[0]:
                if not torch.allclose(model_states[0][key], model_states[i][key]):
                    return

        self.fail("Tous les modèles ont les mêmes poids avant la synchronisation")

    def _assert_models_identical(self):
        """Vérifie que tous les modèles ont les mêmes poids."""
        model_states = []
        for model in self.models:
            state = model.policy.state_dict()
            model_states.append(state)

        # Vérifier que tous les paramètres sont identiques entre les modèles
        for i in range(1, len(model_states)):
            for key in model_states[0]:
                self.assertTrue(
                    torch.allclose(model_states[0][key], model_states[i][key]),
                    f"Les modèles diffèrent sur le paramètre {key}"
                )

    def test_worker_monitor(self):
        """Teste le suivi des performances des workers."""
        # Simuler des trades pour chaque worker
        for i in range(3):
            self.worker_monitor.record_trade(
                worker_id=f"worker_{i}",
                symbol="BTC/USD",
                action="buy",
                price=50000.0 + (i * 1000),  # Prix différents pour chaque worker
                quantity=0.1,
                pnl=10.0 * (i + 1),  # PnL différents pour chaque worker
                pnl_pct=0.2 * (i + 1),
                duration=60.0
            )

        # Vérifier que les statistiques sont correctement mises à jour
        stats = self.worker_monitor.get_stats("worker_1")
        self.assertEqual(stats["total_trades"], 1)
        self.assertEqual(stats["total_pnl"], 20.0)  # worker_1 a un PnL de 20.0

        # Vérifier le récapitulatif
        summary = self.worker_monitor.get_stats()
        self.assertEqual(len(summary), 3)  # 3 workers

        # Vérifier que les PnL sont corrects
        worker_pnls = {worker_id: stats["total_pnl"] for worker_id, stats in summary.items()}
        self.assertEqual(worker_pnls["worker_0"], 10.0)
        self.assertEqual(worker_pnls["worker_1"], 20.0)
        self.assertEqual(worker_pnls["worker_2"], 30.0)


if __name__ == "__main__":
    unittest.main()
