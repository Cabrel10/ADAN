"""
Tests unitaires pour le module model_ensemble.py
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

# Import du module à tester
from adan_trading_bot.model.model_ensemble import (
    ModelEnsemble, 
    ModelPerformance, 
    VotingMechanism
)

# Modèle de test
class DummyModel(nn.Module):
    def __init__(self, output_value=0.5):
        super().__init__()
        self.output_value = output_value
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return torch.ones(x.size(0), 1) * self.output_value


class TestModelPerformance(unittest.TestCase):
    """Tests pour la classe ModelPerformance."""
    
    def setUp(self):
        self.perf = ModelPerformance(
            model_name="test_model",
            weights=1.0,
            metadata={"version": "1.0"}
        )
    
    def test_initial_accuracy(self):
        """Teste la précision initiale."""
        self.assertEqual(self.perf.accuracy, 0.0)
    
    def test_update_performance_correct(self):
        """Teste la mise à jour des performances avec une prédiction correcte."""
        self.perf.update_performance(True, accuracy=0.9)
        self.assertEqual(self.perf.accuracy, 1.0)
        self.assertEqual(self.perf.metadata["accuracy"], 0.9)
    
    def test_update_performance_incorrect(self):
        """Teste la mise à jour des performances avec une prédiction incorrecte."""
        self.perf.update_performance(True)
        self.perf.update_performance(False)
        self.assertEqual(self.perf.accuracy, 0.5)
    
    def test_serialization(self):
        """Teste la sérialisation/désérialisation."""
        self.perf.update_performance(True)
        data = self.perf.to_dict()
        
        # Vérifie que les données sont correctement sérialisées
        self.assertEqual(data["model_name"], "test_model")
        self.assertEqual(data["accuracy"], 1.0)
        
        # Vérifie la désérialisation
        new_perf = ModelPerformance.from_dict(data)
        self.assertEqual(new_perf.model_name, self.perf.model_name)
        self.assertEqual(new_perf.accuracy, self.perf.accuracy)


class TestVotingMechanism(unittest.TestCase):
    """Tests pour la classe VotingMechanism."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.voter = VotingMechanism()
        
        # Crée des prédictions de test
        self.predictions = [
            torch.tensor([[1.0], [1.0], [1.0]], device=self.device),  # Modèle 1
            torch.tensor([[0.0], [0.0], [1.0]], device=self.device),  # Modèle 2
            torch.tensor([[1.0], [0.0], [0.0]], device=self.device),  # Modèle 3
        ]
    
    def test_weighted_voting(self):
        """Teste le vote pondéré."""
        voter = VotingMechanism(method='weighted')
        weights = [0.7, 0.2, 0.1]  # Le premier modèle a plus de poids
        
        result = voter.combine_predictions(self.predictions, weights)
        expected = torch.tensor([[0.7 + 0.0 + 0.1],  # 0.8
                               [0.7 + 0.0 + 0.0],   # 0.7
                               [0.7 + 0.2 + 0.0]],  # 0.9
                              device=self.device)
        
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))
    
    def test_majority_voting(self):
        """Teste le vote majoritaire."""
        voter = VotingMechanism(method='majority')
        
        # Pour [1,0,1] -> 1 (majorité)
        # Pour [1,0,0] -> 0 (majorité)
        # Pour [1,1,0] -> 1 (majorité)
        expected = torch.tensor([[1.0], [0.0], [1.0]], device=self.device)
        
        result = voter.combine_predictions(self.predictions)
        self.assertTrue(torch.equal(result, expected))
    
    def test_average_voting(self):
        """Teste la moyenne des prédictions."""
        voter = VotingMechanism(method='average')
        
        result = voter.combine_predictions(self.predictions)
        expected = torch.tensor([[2/3], [1/3], [2/3]], device=self.device)
        
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))
    
    def test_invalid_method(self):
        """Teste avec une méthode de vote invalide."""
        # Teste que le constructeur lève une ValueError pour une méthode invalide
        with self.assertRaises(ValueError) as context:
            VotingMechanism(method='invalid_method')
            
        # Vérifie que le message d'erreur est correct
        self.assertIn("Méthode de vote non supportée", str(context.exception))


class TestModelEnsemble(unittest.TestCase):
    """Tests pour la classe ModelEnsemble."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp()
        self.performance_file = os.path.join(self.temp_dir, "performance.json")
        
        # Crée un ensemble de modèles de test
        self.ensemble = ModelEnsemble(
            voting_method='weighted',
            performance_file=self.performance_file
        )
        
        # Ajoute des modèles factices
        self.model1 = DummyModel(output_value=0.8)
        self.model2 = DummyModel(output_value=0.2)
        
        self.ensemble.add_model(
            self.model1, 
            "high_confidence_model", 
            initial_weight=0.7,
            version="1.0"
        )
        
        self.ensemble.add_model(
            self.model2, 
            "low_confidence_model", 
            initial_weight=0.3,
            version="1.0"
        )
    
    def tearDown(self):
        # Nettoie les fichiers temporaires
        if os.path.exists(self.performance_file):
            os.remove(self.performance_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_add_remove_model(self):
        """Teste l'ajout et la suppression de modèles."""
        # Vérifie que les modèles ont été ajoutés
        self.assertEqual(len(self.ensemble), 2)
        self.assertIn("high_confidence_model", self.ensemble)
        self.assertIn("low_confidence_model", self.ensemble)
        
        # Teste la suppression d'un modèle
        self.ensemble.remove_model("low_confidence_model")
        self.assertEqual(len(self.ensemble), 1)
        self.assertNotIn("low_confidence_model", self.ensemble)
    
    def test_predict(self):
        """Teste la prédiction avec l'ensemble de modèles."""
        # Crée une entrée factice
        x = torch.randn(3, 10, device=self.device)
        
        # Prédiction attendue: moyenne pondérée des sorties des modèles
        # model1: 0.8 * 0.7 = 0.56
        # model2: 0.2 * 0.3 = 0.06
        # Total: 0.62
        expected = torch.ones(3, 1, device=self.device) * 0.62
        
        result = self.ensemble.predict(x)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))
    
    def test_update_weights(self):
        """Teste la mise à jour des poids des modèles."""
        # Met à jour le poids du premier modèle
        self.ensemble.update_weights("high_confidence_model", 0.9)
        
        # Vérifie que le poids a été mis à jour
        self.assertEqual(self.ensemble.performance["high_confidence_model"].weights, 0.9)
        
        # Vérifie que le poids négatif est ramené à 0
        self.ensemble.update_weights("high_confidence_model", -0.5)
        self.assertEqual(self.ensemble.performance["high_confidence_model"].weights, 0.0)
    
    def test_update_performance(self):
        """Teste la mise à jour des performances des modèles."""
        # Met à jour les performances du premier modèle
        self.ensemble.update_performance("high_confidence_model", True, accuracy=0.85)
        self.ensemble.update_performance("high_confidence_model", False)
        
        # Vérifie que les performances ont été mises à jour
        perf = self.ensemble.performance["high_confidence_model"]
        self.assertEqual(perf.accuracy, 0.5)
        self.assertEqual(perf.metadata["accuracy"], 0.85)
    
    def test_get_best_model(self):
        """Teste l'identification du meilleur modèle."""
        # Met à jour les performances des modèles
        self.ensemble.update_performance("high_confidence_model", True)
        self.ensemble.update_performance("high_confidence_model", False)  # 50%
        
        self.ensemble.update_performance("low_confidence_model", True)
        self.ensemble.update_performance("low_confidence_model", True)    # 100%
        
        # Vérifie que le meilleur modèle est correctement identifié
        best_name, best_acc = self.ensemble.get_best_model()
        self.assertEqual(best_name, "low_confidence_model")
        self.assertEqual(best_acc, 1.0)
    
    def test_save_load_performance(self):
        """Teste la sauvegarde et le chargement des performances."""
        # Met à jour les performances
        self.ensemble.update_performance("high_confidence_model", True)
        self.ensemble.update_performance("high_confidence_model", True)  # 100%
        self.ensemble.update_performance("low_confidence_model", False)  # 0%
        
        # Sauvegarde les performances
        self.ensemble.save_performance()
        
        # Vérifie que le fichier a été créé
        self.assertTrue(os.path.exists(self.performance_file))
        
        # Charge les performances dans un nouvel ensemble
        new_ensemble = ModelEnsemble(performance_file=self.performance_file)
        
        # Ajoute les mêmes modèles au nouvel ensemble
        new_ensemble.add_model(self.model1, "high_confidence_model")
        new_ensemble.add_model(self.model2, "low_confidence_model")
        
        # Vérifie que les performances ont été chargées correctement
        self.assertIn("high_confidence_model", new_ensemble.performance)
        self.assertIn("low_confidence_model", new_ensemble.performance)
        
        # Vérifie les précisions
        self.assertEqual(
            new_ensemble.performance["high_confidence_model"].accuracy,
            1.0,
            "La précision du modèle high_confidence_model devrait être 1.0"
        )
        self.assertEqual(
            new_ensemble.performance["low_confidence_model"].accuracy,
            0.0,
            "La précision du modèle low_confidence_model devrait être 0.0"
        )
        
        # Vérifie également les compteurs de prédictions
        self.assertEqual(
            new_ensemble.performance["high_confidence_model"].total_predictions,
            2,
            "Le nombre total de prédictions pour high_confidence_model devrait être 2"
        )
        self.assertEqual(
            new_ensemble.performance["high_confidence_model"].correct_predictions,
            2,
            "Le nombre de prédictions correctes pour high_confidence_model devrait être 2"
        )
    
    def test_ensemble_with_no_models(self):
        """Teste le comportement avec un ensemble vide."""
        empty_ensemble = ModelEnsemble()
        x = torch.randn(3, 10, device=self.device)
        
        with self.assertRaises(ValueError):
            empty_ensemble.predict(x)


if __name__ == "__main__":
    unittest.main()
