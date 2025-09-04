"""
Tests unitaires pour le module d'optimisation des hyperparamètres.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call

import numpy as np
import optuna
import pytest
from optuna.trial import Trial

from adan_trading_bot.optimization import HyperparameterOptimizer


class TestHyperparameterOptimizer(unittest.TestCase):
    """Tests pour la classe HyperparameterOptimizer."""

    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer une base de données temporaire pour les tests
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.storage_url = f"sqlite:///{self.db_path}"

        # Initialiser l'optimiseur avec la base de données temporaire
        self.optimizer = HyperparameterOptimizer(
            study_name="test_study",
            storage_url=self.storage_url,
            n_trials=10,
            timeout=60,
            n_jobs=1  # Éviter les problèmes de concurrence dans les tests
        )

        # Fonction objectif de test
        def test_objective(trial, x=0):
            a = trial.suggest_float('a', 0, 1)
            b = trial.suggest_float('b', 0, 1)
            return (a - 0.2) ** 2 + (b - 0.3) ** 2 + x

        self.test_objective = test_objective

        # Distribution de paramètres de test
        self.param_distributions = {
            'a': {'type': 'float', 'low': 0, 'high': 1},
            'b': {'type': 'float', 'low': 0, 'high': 1, 'log': True},
            'c': {'type': 'int', 'low': 1, 'high': 10},
            'd': {'type': 'categorical', 'choices': ['relu', 'tanh', 'sigmoid']}
        }

    def tearDown(self):
        """Nettoyage après les tests."""
        # Fermer et supprimer la base de données temporaire
        self.temp_db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_study(self):
        """Teste la création d'une étude Optuna."""
        study = self.optimizer.create_study()
        self.assertIsInstance(study, optuna.study.Study)
        self.assertEqual(study.study_name, "test_study")

    def test_suggest_hyperparameters(self):
        """Teste la suggestion d'hyperparamètres."""
        # Créer un essai factice
        trial = MagicMock(spec=Trial)

        # Configurer les valeurs de retour simulées
        trial.suggest_float.side_effect = [0.5, 0.1]  # a, b
        trial.suggest_int.return_value = 5  # c
        trial.suggest_categorical.return_value = 'relu'  # d

        # Appeler la méthode à tester
        params = self.optimizer.suggest_hyperparameters(trial, self.param_distributions)

        # Vérifier les résultats
        self.assertEqual(params['a'], 0.5)
        self.assertEqual(params['b'], 0.1)
        self.assertEqual(params['c'], 5)
        self.assertEqual(params['d'], 'relu')

        # Vérifier que les méthodes ont été appelées avec les bons arguments
        # Vérifier les appels à suggest_float
        self.assertEqual(trial.suggest_float.call_count, 2, "Doit y avoir exactement 2 appels à suggest_float")

        # Vérifier les appels à suggest_float avec les bons paramètres
        # Pour 'a' (type: float, log=False par défaut)
        trial.suggest_float.assert_any_call(
            'a',
            low=0,
            high=1,
            step=None,
            log=False
        )

        # Pour 'b' (type: float, log=True)
        trial.suggest_float.assert_any_call(
            'b',
            low=0,
            high=1,
            step=None,
            log=True
        )

        # Vérifier l'appel à suggest_int
        trial.suggest_int.assert_called_once_with(
            'c',
            low=1,
            high=10,
            step=1,
            log=False
        )

        # Vérifier l'appel à suggest_categorical
        trial.suggest_categorical.assert_called_once_with(
            'd',
            ['relu', 'tanh', 'sigmoid']
        )

    @patch('optuna.study.Study.optimize')
    def test_optimize(self, mock_optimize):
        """Teste la méthode d'optimisation."""
        # Configurer le mock pour éviter une véritable optimisation
        mock_optimize.return_value = None

        # Appeler la méthode d'optimisation
        study = self.optimizer.optimize(
            objective=self.test_objective,
            param_distributions={
                'a': {'type': 'float', 'low': 0, 'high': 1},
                'b': {'type': 'float', 'low': 0, 'high': 1}
            },
            n_trials=5,
            timeout=30,
            x=10  # Paramètre supplémentaire pour la fonction objectif
        )

        # Vérifier que l'optimisation a été appelée avec les bons paramètres
        self.assertIsNotNone(study)
        mock_optimize.assert_called_once()

        # Vérifier les arguments de l'appel à optimize
        args, kwargs = mock_optimize.call_args
        self.assertEqual(kwargs['n_trials'], 5)
        self.assertEqual(kwargs['timeout'], 30)
        self.assertEqual(kwargs['n_jobs'], 1)

    def test_get_best_params(self):
        """Teste la récupération des meilleurs paramètres."""
        # Créer une étude de test avec un essai
        study = self.optimizer.create_study()
        study.optimize(lambda t: (t.suggest_float('x', 0, 1) - 0.5) ** 2, n_trials=1)

        # Récupérer les meilleurs paramètres
        best_params = self.optimizer.get_best_params(study)

        # Vérifier que les paramètres sont bien renvoyés
        self.assertIn('x', best_params)
        self.assertIsInstance(best_params['x'], float)

    def test_get_best_trial(self):
        """Teste la récupération du meilleur essai."""
        # Créer une étude de test avec un essai
        study = self.optimizer.create_study()
        study.optimize(lambda t: (t.suggest_float('x', 0, 1) - 0.5) ** 2, n_trials=1)

        # Récupérer le meilleur essai
        best_trial = self.optimizer.get_best_trial(study)

        # Vérifier que l'essai est bien renvoyé
        self.assertIsNotNone(best_trial)
        self.assertIn('x', best_trial.params)
        self.assertIsInstance(best_trial.params['x'], float)


if __name__ == '__main__':
    unittest.main()
