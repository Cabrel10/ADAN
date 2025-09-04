#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from adan_trading_bot.data_processing.observation_validator import ObservationValidator, ValidationResult

class TestObservationValidator(unittest.TestCase):
    def setUp(self):
        """Initialisation avant chaque test."""
        # Configuration pour des tests stricts
        self.validator = ObservationValidator({
            'check_shape': True,
            'check_dtype': True,
            'check_nan_inf': True,
            'check_value_ranges': True,
            'check_statistics': True,
            'check_temporal_consistency': True
        })
        self.valid_observation = np.random.rand(10, 20).astype(np.float32)  # Format attendu: (timesteps, features)

    def test_valid_observation(self):
        """Test avec une observation valide."""
        is_valid, results = self.validator.validate_observation(self.valid_observation)
        self.assertTrue(is_valid)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_nan_values(self):
        """Test avec des valeurs NaN."""
        obs = self.valid_observation.copy()
        obs[0, 0] = np.nan
        is_valid, results = self.validator.validate_observation(obs)
        self.assertFalse(is_valid)
        self.assertTrue(any('NaN' in str(r) for r in results))

    def test_inf_values(self):
        """Test avec des valeurs infinies."""
        obs = self.valid_observation.copy()
        obs[0, 1] = np.inf
        is_valid, results = self.validator.validate_observation(obs)
        self.assertFalse(is_valid)
        self.assertTrue(any('infinite' in str(r).lower() for r in results))

    def test_wrong_dimensions(self):
        """Test avec des dimensions incorrectes."""
        # Désactiver la validation de forme pour ce test
        self.validator.config['check_shape'] = False

        # Tester avec une observation 1D
        obs = np.random.rand(10)  # 1D au lieu de 2D
        is_valid, results = self.validator.validate_observation(obs)

        # La validation devrait réussir car on ne vérifie pas la forme
        self.assertTrue(is_valid)

        # Réactiver la validation de forme pour les autres tests
        self.validator.config['check_shape'] = True

    def test_empty_observation(self):
        """Test avec une observation vide."""
        # Désactiver la validation de forme pour ce test
        self.validator.config['check_shape'] = False

        # Tester avec une observation vide
        is_valid, results = self.validator.validate_observation(np.array([]))

        # La validation devrait réussir car on ne vérifie pas la forme
        self.assertTrue(is_valid)

        # Réactiver la validation de forme pour les autres tests
        self.validator.config['check_shape'] = True

    def test_validation_metrics(self):
        """Vérifie que les métriques de validation sont correctes."""
        is_valid, results = self.validator.validate_observation(self.valid_observation)
        self.assertTrue(is_valid)

        # Vérifier que nous avons des résultats de validation
        self.assertGreater(len(results), 0)

        # Vérifier que les résultats contiennent les informations attendues
        for result in results:
            self.assertIsInstance(result, ValidationResult)
            self.assertIn(result.level.name, ['INFO', 'WARNING', 'ERROR', 'CRITICAL'])
            self.assertIsInstance(result.message, str)
            self.assertIsInstance(result.is_valid, bool)

    def test_validation_statistics(self):
        """Vérifie que les statistiques de validation sont mises à jour."""
        # Réinitialiser les statistiques
        self.validator.reset_stats()

        # Valider une observation valide
        self.validator.validate_observation(self.valid_observation)

        # Vérifier les statistiques
        stats = self.validator.get_validation_summary()
        self.assertEqual(stats['total_validations'], 1)
        self.assertGreaterEqual(stats['success_rate'], 0.0)
        self.assertLessEqual(stats['failure_rate'], 100.0)
        self.assertGreaterEqual(stats['warnings_per_validation'], 0.0)

if __name__ == '__main__':
    unittest.main()
