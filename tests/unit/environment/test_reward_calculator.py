#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests unitaires pour le calculateur de récompense adaptatif."""
import unittest
import logging
from adan_trading_bot.environment.reward_calculator import (
    AdaptiveRewardCalculator,
    MarketRegime
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)

class TestMarketRegimeDetector:
    """Classe de test pour le détecteur de régime de marché."""

    def __init__(self, regime=MarketRegime.RANGING, strength=0.5, volatility=0.1):
        self.regime = regime
        self.strength = strength
        self.volatility = volatility

    def update(self, price: float) -> None:
        """Mise à jour factice du détecteur."""
        pass

    def get_regime(self) -> MarketRegime:
        """Retourne le régime simulé."""
        return self.regime

    def get_regime_strength(self) -> float:
        """Retourne la force du régime simulée."""
        return self.strength

    def get_volatility(self) -> float:
        """Retourne la volatilité simulée."""
        return self.volatility


class TestAdaptiveRewardCalculator(unittest.TestCase):
    def setUp(self):
        # Crée un détecteur de test avec des paramètres spécifiques
        self.detector = TestMarketRegimeDetector(
            regime=MarketRegime.RANGING,
            strength=0.5,
            volatility=0.1
        )

        # Initialise le calculateur avec des paramètres de test
        self.calculator = AdaptiveRewardCalculator(
            lookback_period=14,
            volatility_threshold=0.02,
            trend_strength_threshold=0.6,
            min_data_points=5
        )

        # Remplace le détecteur interne par notre mock
        self.calculator.regime_detector = self.detector

        # Initialise les attributs nécessaires
        self.calculator.current_regime = MarketRegime.RANGING
        self.calculator.regime_strength = 0.5
        self.calculator.position_size = 0.0
        self.calculator.inaction_penalty = -0.1
        self.calculator.commission_penalty = 1.0
        self.calculator.min_profit_multiplier = 1.0
        self.calculator.optimal_trade_bonus = 0.05
        self.calculator.clipping_range = (-10.0, 10.0)

    def test_smooth_regime_transition(self):
        """Teste la transition progressive entre les régimes."""
        # Configuration initiale
        self.detector.regime = MarketRegime.RANGING
        self.detector.strength = 0.5

        # Mise à jour initiale
        self.calculator.update_market_regime(100.0)
        initial_penalty = self.calculator.inaction_penalty

        # Changement vers un régime de tendance haussière
        self.detector.regime = MarketRegime.TRENDING
        self.detector.strength = 0.9

        # Mise à jour progressive (simule plusieurs étapes)
        for i in range(5):
            self.calculator.update_market_regime(100.0 + i)

        # Vérification que le régime a changé
        self.assertEqual(self.calculator.current_regime, MarketRegime.TRENDING)

        # Vérification que la pénalité a été mise à jour
        self.assertNotEqual(self.calculator.inaction_penalty, initial_penalty)

    def test_optimal_trade_bonus_calculation(self):
        """Teste le calcul du bonus pour les trades optimaux."""
        # Configuration pour un régime de tendance
        self.detector.regime = MarketRegime.TRENDING
        self.detector.strength = 0.9

        # Mise à jour du régime
        self.calculator.update_market_regime(100.0)

        # Trade avec un bon ratio profit/commission (supérieur à min_profit_multiplier)
        reward1 = self.calculator.calculate(
            current_price=100.0,
            realized_pnl=20.0,  # Gros profit
            unrealized_pnl=0.0,
            commission=1.0,     # Faible commission
            position_size=1.0
        )

        # Trade avec un mauvais ratio profit/commission (inférieur à min_profit_multiplier)
        reward2 = self.calculator.calculate(
            current_price=100.0,
            realized_pnl=0.5,   # Petit profit
            unrealized_pnl=0.0,
            commission=1.0,     # Même commission
            position_size=1.0
        )

        # Vérification que les récompenses sont dans les limites attendues
        self.assertGreater(reward1, -10.0)
        self.assertLess(reward1, 10.0)
        self.assertGreater(reward2, -10.0)
        self.assertLess(reward2, 10.0)

        # Vérification que le bonus a été appliqué pour le premier trade
        self.assertGreater(reward1, reward2)

    def test_reward_adaptation_volatile(self):
        """Teste l'adaptation des récompenses en régime de forte volatilité."""
        # Configuration pour un régime volatil
        self.detector.regime = MarketRegime.VOLATILE
        self.detector.strength = 0.9

        # Mise à jour du régime
        self.calculator.update_market_regime(100.0)

        # Calcul d'une récompense avec un profit
        reward = self.calculator.calculate(
            current_price=100.0,
            realized_pnl=5.0,
            unrealized_pnl=0.0,
            commission=0.5,
            position_size=1.0
        )

        # Vérification que la récompense est dans la plage attendue
        self.assertGreaterEqual(reward, -10.0)  # Doit être supérieur ou égal à la limite inférieure de clipping
        self.assertLessEqual(reward, 10.0)     # Doit être inférieur ou égal à la limite supérieure de clipping

    def test_smooth_transition(self):
        """Teste la transition progressive des paramètres entre les régimes."""
        # Régime initial
        self.detector.regime = MarketRegime.RANGING
        self.detector.strength = 0.5
        self.calculator.update_market_regime(100.0)
        initial_penalty = self.calculator.inaction_penalty

        # Changement vers un nouveau régime avec une force partielle
        self.detector.regime = MarketRegime.TRENDING
        self.detector.strength = 0.3

        # Mise à jour du régime
        self.calculator.update_market_regime(100.0)

        # Vérification que la pénalité a changé
        self.assertNotEqual(self.calculator.inaction_penalty, initial_penalty)

        # Vérification que la pénalité est dans la plage attendue pour le nouveau régime
        self.assertLessEqual(self.calculator.inaction_penalty, -0.15)  # Valeur pour le régime TRENDING
        self.assertGreaterEqual(self.calculator.inaction_penalty, -0.2)  # Valeur pour le régime RANGING

        # Transition complète
        self.detector.strength = 1.0
        self.calculator.update_market_regime(100.0)

        # Vérifie que la pénalité a évolué dans la bonne direction
        self.assertLess(self.calculator.inaction_penalty, initial_penalty)  # La pénalité devrait diminuer pour TRENDING


if __name__ == '__main__':
    unittest.main()
