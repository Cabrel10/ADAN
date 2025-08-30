#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le calculateur de récompense adaptatif.
"""

import unittest
import numpy as np
import logging
from adan_trading_bot.environment.reward_calculator import RewardCalculator

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestRewardCalculator(unittest.TestCase):
    """Tests pour le calculateur de récompense adaptatif."""
    
    def setUp(self):
        # Configuration minimale pour les tests
        self.config = {
            'reward_shaping': {
                'realized_pnl_multiplier': 1.0,
                'unrealized_pnl_multiplier': 0.1,
                'inaction_penalty': -0.1,  # Corrigé pour correspondre au test
                'commission_penalty': 1.5,
                'min_profit_multiplier': 3.0,
                'optimal_trade_bonus': 1.0,
                'performance_threshold': 0.8,
                'regime_lookback': 5,
                'volatility_threshold': 0.02,
                'reward_clipping_range': [-50.0, 50.0]  # Increased range to allow higher rewards
            }
        }
        self.calculator = RewardCalculator(self.config)
    
    def test_initialization(self):
        """Vérifie l'initialisation du calculateur."""
        # Le RewardCalculator n'utilise plus directement MarketRegimeDetector
        # Vérifions simplement que l'initialisation fonctionne
        self.assertEqual(self.calculator.pnl_multiplier, 1.0)
        self.assertEqual(self.calculator.inaction_penalty, -0.1)
    
    def test_calculate_reward_profitable_trade(self):
        """Teste le calcul de récompense pour un trade rentable."""
        # Données de test
        portfolio_metrics = {
            'total_commission': 1.0,
            'drawdown': 0.0,
            'sharpe_ratio': 1.0
        }
        trade_pnl = 10.0
        action = 1  # Buy action
        
        # Appel à la méthode calculate
        reward = self.calculator.calculate(
            portfolio_metrics=portfolio_metrics,
            trade_pnl=trade_pnl,
            action=action
        )
        
        # Vérifie que la récompense est positive
        self.assertGreater(reward, 0)
        # Vérifie que la récompense est cohérente avec les paramètres
        expected_min_reward = (trade_pnl - portfolio_metrics['total_commission'] * self.calculator.commission_penalty) * self.calculator.pnl_multiplier
        self.assertGreaterEqual(reward, expected_min_reward)
    
    def test_calculate_reward_unprofitable_trade(self):
        """Teste le calcul de récompense pour un trade non rentable."""
        # Données de test
        portfolio_metrics = {
            'total_commission': 1.0,
            'drawdown': 0.0,
            'sharpe_ratio': 1.0
        }
        trade_pnl = -5.0  # Perte de 5.0
        action = 2  # Sell action
        
        # Appel à la méthode calculate
        reward = self.calculator.calculate(
            portfolio_metrics=portfolio_metrics,
            trade_pnl=trade_pnl,
            action=action
        )
        
        # Vérifie que la récompense est négative
        self.assertLess(reward, 0)
        # Vérifie que la pénalité est cohérente
        # Calculate base expected reward without Sharpe ratio bonus
        base_expected_reward = (trade_pnl - portfolio_metrics['total_commission'] * self.calculator.commission_penalty) * self.calculator.pnl_multiplier
        
        # Add maximum possible Sharpe ratio bonus (sharpe_ratio=1.0 * 0.1 = 0.1)
        max_sharpe_bonus = 1.0 * 0.1  # sharpe_ratio=1.0 * 0.1
        expected_max_reward = base_expected_reward + max_sharpe_bonus
        
        # Add a small epsilon to account for floating point arithmetic
        self.assertLessEqual(reward, expected_max_reward + 1e-9, 
                           f"Expected reward <= {expected_max_reward}, but got {reward}")
    
    def test_inaction_penalty(self):
        """Teste l'application de la pénalité d'inaction."""
        # Données de test pour une action de maintien (inaction)
        portfolio_metrics = {
            'total_commission': 0.0,
            'drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        trade_pnl = 0.0  # Aucun trade fermé
        action = 0  # Action de maintien
        
        # Appel à la méthode calculate
        reward = self.calculator.calculate(
            portfolio_metrics=portfolio_metrics,
            trade_pnl=trade_pnl,
            action=action
        )
        
        # Vérifie que la pénalité d'inaction est appliquée
        self.assertEqual(reward, self.calculator.inaction_penalty)
    
    def test_optimal_trade_bonus(self):
        """Teste l'application du bonus pour les trades optimaux."""
        # Données de test pour un trade très rentable
        portfolio_metrics = {
            'total_commission': 1.0,
            'drawdown': 0.0,
            'sharpe_ratio': 2.0
        }
        trade_pnl = 10.0  # Profit élevé
        action = 1  # Buy action
        chunk_id = 1
        optimal_chunk_pnl = 15.0
        performance_ratio = 0.9  # 90% de l'optimal
        
        # Appel à la méthode calculate avec des paramètres pour déclencher un bonus
        reward = self.calculator.calculate(
            portfolio_metrics=portfolio_metrics,
            trade_pnl=trade_pnl,
            action=action,
            chunk_id=chunk_id,
            optimal_chunk_pnl=optimal_chunk_pnl,
            performance_ratio=performance_ratio
        )
        
        # Vérifie que la récompense est supérieure à la récompense de base (sans bonus)
        base_reward = (trade_pnl - portfolio_metrics['total_commission'] * self.calculator.commission_penalty) * self.calculator.pnl_multiplier
        self.assertGreater(reward, base_reward)
    
    def test_sharpe_ratio_calculation(self):
        """Teste le calcul du ratio de Sharpe."""
        # Réinitialiser le calculateur avec une configuration spécifique
        self.calculator = RewardCalculator(self.config)
        
        # Ajouter des retours historiques pour le calcul
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        for r in returns:
            self.calculator._update_returns_history(r)
        
        # Calculer le ratio de Sharpe
        sharpe_ratio = self.calculator._calculate_sharpe_ratio()
        
        # Vérifier que le ratio de Sharpe est un nombre flottant
        self.assertIsInstance(sharpe_ratio, float)
        # Le ratio de Sharpe devrait être positif pour ces retours
        self.assertGreater(sharpe_ratio, 0.0)
    
    def test_sortino_ratio_calculation(self):
        """Teste le calcul du ratio de Sortino."""
        # Réinitialiser le calculateur avec une configuration spécifique
        self.calculator = RewardCalculator(self.config)
        
        # Ajouter des retours historiques pour le calcul
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        for r in returns:
            self.calculator._update_returns_history(r)
        
        # Calculer le ratio de Sortino
        sortino_ratio = self.calculator._calculate_sortino_ratio()
        
        # Vérifier que le ratio de Sortino est un nombre flottant
        self.assertIsInstance(sortino_ratio, float)
        # Le ratio de Sortino devrait être positif pour ces retours
        self.assertGreater(sortino_ratio, 0.0)
    
    def test_calmar_ratio_calculation(self):
        """Teste le calcul du ratio de Calmar."""
        # Réinitialiser le calculateur avec une configuration spécifique
        self.calculator = RewardCalculator(self.config)
        
        # Ajouter des retours historiques pour le calcul
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        for r in returns:
            self.calculator._update_returns_history(r)
        
        # Données de portefeuille avec un drawdown maximum
        portfolio_metrics = {
            'max_drawdown': 0.05  # 5% de drawdown maximum
        }
        
        # Calculer le ratio de Calmar
        calmar_ratio = self.calculator._calculate_calmar_ratio(portfolio_metrics)
        
        # Vérifier que le ratio de Calmar est un nombre flottant
        self.assertIsInstance(calmar_ratio, float)
    
    def test_composite_score_calculation(self):
        """Teste le calcul du score composite."""
        # Réinitialiser le calculateur avec une configuration spécifique
        self.calculator = RewardCalculator(self.config)
        
        # Ajouter des retours historiques pour le calcul des ratios
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        for r in returns:
            self.calculator._update_returns_history(r)
        
        # Données de portefeuille pour le test
        portfolio_metrics = {
            'total_commission': 1.0,
            'drawdown': 0.02,
            'max_drawdown': 0.05,
            'sharpe_ratio': 1.5
        }
        
        # Appel à la méthode calculate pour déclencher le calcul du score composite
        reward = self.calculator.calculate(
            portfolio_metrics=portfolio_metrics,
            trade_pnl=10.0,
            action=1  # Buy action
        )
        
        # Vérifier que la récompense est un nombre flottant
        self.assertIsInstance(reward, float)
        # Vérifier que la récompense est dans la plage de clipping
        self.assertGreaterEqual(reward, self.calculator.clipping_range[0])
        self.assertLessEqual(reward, self.calculator.clipping_range[1])


if __name__ == '__main__':
    unittest.main()
