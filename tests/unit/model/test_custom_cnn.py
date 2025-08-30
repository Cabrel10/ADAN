"""Tests unitaires pour le module CustomCNN."""
import unittest
from unittest.mock import patch, MagicMock
from torch import nn
import torch

from adan_trading_bot.model.custom_cnn import CustomCNN


class TestCustomCNN(unittest.TestCase):
    """Tests pour la classe CustomCNN."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.observation_space = MagicMock()
        self.observation_space.shape = (3, 84, 84)  # Format CHW
        # Ajuster la dimension de sortie attendue à 128 pour correspondre à la configuration par défaut
        self.features_dim = 128
        self.batch_size = 4
        
        # Créer un modèle de test
        self.model = CustomCNN(
            observation_space=self.observation_space,
            features_dim=self.features_dim,
            cnn_configs={
                'block_a': {
                    'out_channels': 32,
                    'kernel_size': 3,
                    'padding': 1,
                    'leaky_relu_negative_slope': 0.01,
                    'dropout': 0.1
                },
                'block_b': {
                    'multi_scale': [
                        {'kernel_size': 3, 'dilation': 1, 'padding': 1},
                        {'kernel_size': 5, 'dilation': 1, 'padding': 2},
                    ],
                    'se_ratio': 8,
                    'dropout': 0.1,
                    'leaky_relu_negative_slope': 0.01
                },
                'attention': {
                    'num_heads': 4,
                    'dropout': 0.1,
                    'use_residual': True
                },
                'head': {
                    'hidden_units': [128],
                    'dropout': 0.2,
                    'activation': 'leaky_relu'
                }
            },
            memory_config={
                'enable_memory_efficient': True,
                'enable_mixed_precision': True,
                'enable_gradient_checkpointing': False,
                'aggressive_cleanup': False
            }
        )
        
        # Créer des données d'entrée factices
        self.dummy_input = torch.randn(
            self.batch_size, 
            *self.observation_space.shape
        )
    
    def test_forward_pass(self):
        """Test du passage avant du modèle."""
        # Act
        output = self.model(self.dummy_input)
        
        # Assert
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (self.batch_size, self.features_dim))
    
    @patch('torch.compile')
    def test_model_compilation(self, mock_compile):
        """Test de la compilation du modèle."""
        # Arrange
        mock_compile.return_value = self.model._forward_impl
        
        # Act
        self.model._maybe_compile()
        
        # Assert
        if torch.__version__ >= '2.0.0' and torch.cuda.is_available():
            mock_compile.assert_called_once()
            self.assertTrue(self.model._compiled)
    
    def test_optimize_for_inference(self):
        """Test de l'optimisation pour l'inférence."""
        # Act
        self.model.optimize_for_inference()
        
        # Assert
        self.assertFalse(self.model.training)
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                self.assertEqual(module.p, 0.0)

    def test_mixed_precision(self):
        """Test du mode de précision mixte."""
        # Act
        self.model.enable_mixed_precision()
        with torch.amp.autocast(device_type='cuda', enabled=True):
            output = self.model(self.dummy_input)
        
        # Assert
        self.assertIsInstance(output, torch.Tensor)
        # Vérifier que le modèle est en mode précision mixte
        self.assertTrue(self.model._mixed_precision)
        # La sortie peut être en float16 ou float32 selon l'implémentation
        self.assertIn(output.dtype, [torch.float16, torch.float32])


if __name__ == '__main__':
    unittest.main()
