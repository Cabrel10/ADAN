#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le Dynamic Behavior Engine (DBE) adaptatif.
"""

import unittest
import numpy as np
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

from adan_trading_bot.environment.adaptive_dbe import (
    AdaptiveDBE, MarketRegime, DBEParameters, MarketRegimeDetector
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAdaptiveDBE(unittest.TestCase):
    """Tests pour le Dynamic Behavior Engine adaptatif."""

    def setUp(self):
        """Initialisation avant chaque test."""
        # Création d'un mock pour le détecteur de régime
        self.mock_detector = MagicMock(spec=MarketRegimeDetector)
        self.mock_detector.get_ewma_volatility.return_value = 0.02  # 2% de volatilité
        self.mock_detector.detect_regime.return_value = MarketRegime.LOW_VOLATILITY
        self.mock_detector.get_market_conditions.return_value = {
            'volatility': 0.02,
            'trend_strength': 0.0,
            'liquidity': 1000000.0,
            'spread': 0.0005
        }

        # Configuration des paramètres de test
        self.params = DBEParameters(
            volatility_threshold_low=0.01,
            volatility_threshold_high=0.05,
            max_drawdown_threshold=0.15
        )

        # Création de l'instance DBE avec le mock
        self.patcher = patch('adan_trading_bot.environment.adaptive_dbe.MarketRegimeDetector',
                           return_value=self.mock_detector)
        self.MockMarketRegimeDetector = self.patcher.start()

        self.dbe = AdaptiveDBE(
            initial_params=self.params,
            adaptation_enabled=True,
            save_path="test_logs/dbe",
            tau=0.03,
            ewma_lambda=0.94
        )

    def tearDown(self):
        """Nettoyage après chaque test."""
        self.patcher.stop()

    def test_initialization(self):
        """Teste l'initialisation du DBE."""
        self.assertIsNotNone(self.dbe)
        self.assertTrue(self.dbe.adaptation_enabled)
        self.assertEqual(self.dbe.tau, 0.03)
        self.assertIsInstance(self.dbe.params, DBEParameters)

    def test_update_with_market_data(self):
        """Teste la mise à jour avec des données de marché."""
        # Configuration du mock pour retourner un régime de marché spécifique
        self.mock_detector.detect_regime.return_value = MarketRegime.LOW_VOLATILITY
        self.mock_detector.get_ewma_volatility.return_value = 0.015

        # Données de marché simulées
        market_data = {
            'price': 100.0,
            'volume': 1000.0,
            'volatility': 0.015,
            'current_drawdown': 0.05,  # 5% de drawdown (float, pas un mock)
            'trend_strength': 0.3,     # Tendance modérée
            'returns': 0.001,           # Retour de 0.1%
            'liquidity': 1000000.0,    # Liquidité élevée
            'spread': 0.0005,          # Spread de 0.05%
            'market_conditions': {
                'volatility': 0.015,
                'trend_strength': 0.3,
                'liquidity': 1000000.0,
                'spread': 0.0005
            }
        }

        # Métriques de performance simulées
        performance_metrics = {
            'sharpe_ratio': 1.5,
            'sortino_ratio': 1.8,
            'max_drawdown': 0.1,
            'win_rate': 0.6
        }

        # Appel de la méthode update
        modulation = self.dbe.update(market_data, performance_metrics)

        # Vérifications
        self.assertIsNotNone(modulation)
        self.assertIn('risk_level', modulation)
        self.assertIn('position_size', modulation)
        self.assertIn('stop_loss', modulation)
        self.assertIn('take_profit', modulation)
        self.assertIn('volatility', modulation)

        # Vérification des limites des valeurs de modulation
        self.assertGreaterEqual(modulation['risk_level'], 0.1)
        self.assertLessEqual(modulation['risk_level'], 2.0)
        self.assertGreaterEqual(modulation['position_size'], 0.2)
        self.assertLessEqual(modulation['position_size'], 2.0)
        self.assertLessEqual(modulation['stop_loss'], 0.10)  # Max 10%
        self.assertLessEqual(modulation['take_profit'], 0.20)  # Max 20%

    def test_high_volatility_regime(self):
        """Teste le comportement en régime de haute volatilité."""
        # Configuration pour un régime de haute volatilité
        self.mock_detector.detect_regime.return_value = MarketRegime.HIGH_VOLATILITY
        self.mock_detector.get_ewma_volatility.return_value = 0.08

        # Données de marché simulées
        market_data = {
            'price': 100.0,
            'volume': 2000.0,
            'volatility': 0.08,
            'current_drawdown': 0.12,
            'trend_strength': 0.1
        }

        performance_metrics = {
            'sharpe_ratio': 0.8,
            'sortino_ratio': 0.9,
            'max_drawdown': 0.15,
            'win_rate': 0.55
        }

        # Appel de la méthode update
        modulation = self.dbe.update(market_data, performance_metrics)

        # En haute volatilité, on s'attend à une réduction de la taille de position
        # et des stop-loss plus larges
        self.assertLess(modulation['position_size'], 1.0)  # Position réduite
        self.assertGreater(modulation['stop_loss'], 0.02)  # Stop-loss plus large

    def test_crisis_regime(self):
        """Teste le comportement en période de crise."""
        # Configuration pour un régime de crise
        self.mock_detector.detect_regime.return_value = MarketRegime.CRISIS
        self.mock_detector.get_ewma_volatility.return_value = 0.15

        # Données de marché simulées (conditions extrêmes)
        market_data = {
            'price': 100.0,
            'volume': 5000.0,
            'volatility': 0.15,    # Volatilité très élevée
            'current_drawdown': 0.25,  # Drawdown important (float, pas un mock)
            'trend_strength': -0.2,    # Forte tendance baissière
            'returns': -0.05,      # Pertes de 5%
            'liquidity': 500000.0, # Liquidité réduite
            'spread': 0.002,      # Spread élargi
            'market_conditions': {
                'volatility': 0.15,
                'trend_strength': -0.2,
                'liquidity': 500000.0,
                'spread': 0.002
            }
        }

        performance_metrics = {
            'sharpe_ratio': -1.0,
            'sortino_ratio': -1.2,
            'max_drawdown': 0.25,
            'win_rate': 0.4
        }

        # Appel de la méthode update
        modulation = self.dbe.update(market_data, performance_metrics)

        # En période de crise, on s'attend à une réduction drastique du risque
        self.assertLess(modulation['risk_level'], 0.5)  # Risque fortement réduit
        self.assertLess(modulation['position_size'], 0.5)  # Position très réduite
        self.assertGreater(modulation['stop_loss'], 0.05)  # Stop-loss très large

if __name__ == '__main__':
    unittest.main()
