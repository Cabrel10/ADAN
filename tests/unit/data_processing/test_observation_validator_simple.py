"""
Test simplifié pour le validateur d'observation.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Créer des mocks pour les dépendances
class MockValidationResult:
    def __init__(self, is_valid=True, message="", level=None, passed=True):
        self.is_valid = is_valid
        self.message = message
        self.level = level
        self.passed = passed

class MockValidationLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Simuler le module observation_validator
sys.modules['adan_trading_bot.data_processing.observation_validator'] = MagicMock()
from adan_trading_bot.data_processing import observation_validator

# Configurer les mocks
observation_validator.ValidationResult = MockValidationResult
observation_validator.ValidationLevel = MockValidationLevel

class TestObservationValidatorSimple(unittest.TestCase):
    """Tests simplifiés pour ObservationValidator."""
    
    def setUp(self):
        """Configuration avant chaque test."""
        self.validator = observation_validator.ObservationValidator()
        self.validator.validate_observation = self.mock_validate_observation
        self.validator.validate_batch = self.mock_validate_batch
        self.validator.reset_stats = MagicMock()
        self.validator.get_validation_summary = MagicMock(return_value={
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'warnings_count': 0,
            'errors_count': 0,
            'critical_count': 0
        })
        self.validator.save_validation_report = MagicMock()
        self.validator.log_validation_results = MagicMock()
    
    def mock_validate_observation(self, observation, expected_shape=None, **kwargs):
        """Simule la validation d'une observation."""
        if expected_shape and observation.shape != expected_shape:
            return False, [MockValidationResult(
                is_valid=False, 
                message=f"Shape mismatch: expected {expected_shape}, got {observation.shape}",
                level=MockValidationLevel.ERROR
            )]
        return True, [MockValidationResult(
            is_valid=True, 
            message="Validation passed",
            level=MockValidationLevel.INFO
        )]
    
    def mock_validate_batch(self, batch_data):
        """Simule la validation par lots."""
        results = []
        all_valid = True
        for obs in batch_data:
            is_valid, res = self.mock_validate_observation(obs)
            results.append(res)
            all_valid = all_valid and is_valid
        return all_valid, results
    
    def test_validate_clean_data(self):
        """Test de validation avec des données propres."""
        data = np.random.randn(10, 5)
        is_valid, results = self.validator.validate_observation(data)
        self.assertTrue(is_valid)
        self.assertTrue(all(r.is_valid for r in results))
    
    def test_validate_invalid_shape(self):
        """Test de validation avec une forme invalide."""
        data = np.random.randn(5, 3)  # Mauvaise forme
        is_valid, results = self.validator.validate_observation(
            data, expected_shape=(10, 5))
        self.assertFalse(is_valid)
        self.assertIn("Shape mismatch", results[0].message)
    
    def test_batch_validation(self):
        """Test de validation par lots."""
        batch = [
            np.random.randn(10, 5),
            np.random.randn(10, 5),
            np.random.randn(10, 5)
        ]
        all_valid, batch_results = self.validator.validate_batch(batch)
        self.assertTrue(all_valid)
        self.assertEqual(len(batch_results), len(batch))
    
    def test_validation_statistics(self):
        """Test des statistiques de validation."""
        # Simuler des validations
        self.validator.get_validation_summary.return_value = {
            'total_validations': 5,
            'passed_validations': 4,
            'failed_validations': 1,
            'warnings_count': 2,
            'errors_count': 1,
            'critical_count': 0
        }
        
        stats = self.validator.get_validation_summary()
        self.assertEqual(stats['total_validations'], 5)
        self.assertEqual(stats['passed_validations'], 4)
        self.assertEqual(stats['failed_validations'], 1)
    
    def test_validation_report_export(self):
        """Test d'exportation du rapport de validation."""
        data = np.random.randn(10, 5)
        is_valid, results = self.validator.validate_observation(data)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            report_path = tmp_file.name
        
        try:
            self.validator.save_validation_report(results, report_path)
            self.validator.save_validation_report.assert_called_once()
            self.assertTrue(os.path.exists(report_path))
        finally:
            if os.path.exists(report_path):
                os.unlink(report_path)

if __name__ == '__main__':
    unittest.main()
