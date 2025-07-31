#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from adan_trading_bot.environment.action_translator import ActionTranslator
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

class TestActionTranslator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        cls.assets = ['BTCUSDT', 'ETHUSDT']
        cls.current_prices = {'BTCUSDT': 10000, 'ETHUSDT': 500}
        
    def setUp(self):
        """Réinitialisation avant chaque test."""
        self.translator = ActionTranslator(self.assets)
        self.capital_tiers = [
            {
                'name': 'test_tier_1',
                'min_capital': 0,
                'max_position_size_pct': 0.5,
                'risk_per_trade_pct': 1.0,
                'max_drawdown_pct': 5.0,
                'leverage': 1.0
            },
            {
                'name': 'test_tier_2',
                'min_capital': 10000,
                'max_position_size_pct': 0.7,
                'risk_per_trade_pct': 1.5,
                'max_drawdown_pct': 4.0,
                'leverage': 1.0
            }
        ]
        self.portfolio = PortfolioManager({
            'environment': {
                'initial_balance': 10000.0,  # Capital plus élevé pour les tests
                'assets': self.assets
            },
            'trading_rules': {
                'commission_pct': 0.001,
                'futures_enabled': False,
                'min_trade_size': 0.0001,
                'min_notional_value': 10.0,
                'max_notional_value': 1000000.0
            },
            'risk_management': {
                'capital_tiers': self.capital_tiers,
                'position_sizing': {
                    'concentration_limits': {
                        'max_single_asset': 0.5
                    }
                }
            }
        })

    def test_buy_action_single_asset(self):
        """Test d'un achat simple sur un seul actif."""
        # Augmenter légèrement le capital pour utiliser le deuxième palier (0.7)
        self.portfolio.initial_capital = 10001.0
        
        # Action > 0.5 déclenche un achat
        action = np.array([0.6, 0.0])  # Achat sur BTC (0.6 > 0.5), neutre sur ETH
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['asset'], 'BTCUSDT')
        self.assertGreater(orders[0]['units'], 0)
        
        # Vérification que le montant est cohérent avec max_position_size_pct (0.7 pour ce palier)
        # La taille de la position est calculée comme : capital * position_size_pct / prix
        # Dans ce cas, on s'attend à ce que la taille de la position soit de 0.7 BTC (7000.7$ / 10000$)
        expected_units = 0.7  # (10001 * 0.7) / 10000 ≈ 0.7
        self.assertAlmostEqual(orders[0]['units'], expected_units, delta=0.01)
        
        # Test avec une action en dessous du seuil (ne doit pas déclencher d'achat)
        action = np.array([0.4, 0.0])
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        self.assertEqual(len(orders), 0)

    def test_sell_action(self):
        """Test d'une vente avec position existante."""
        # Créer une position initiale
        position_size = 1.0  # 1 BTC de position
        self.portfolio.positions['BTCUSDT'].open(
            entry_price=9000.0,
            size=position_size,
            stop_loss_pct=0.01,
            take_profit_pct=0.02
        )
        
        # Vente partielle (action < -0.5 déclenche une vente)
        # La logique actuelle utilise la valeur absolue du signal pour le pourcentage à vendre
        action = np.array([-0.6, 0.0])  # Vendre 60% de la position
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['asset'], 'BTCUSDT')
        
        # Vérifier que 60% de la position est vendue (valeur absolue de -0.6)
        # La logique actuelle utilise la valeur absolue du signal pour le pourcentage à vendre
        expected_units = -(position_size * 0.6)  # 60% de 1.0 = -0.6
        self.assertAlmostEqual(orders[0]['units'], expected_units, delta=0.001)
        
        # Mettre à jour la taille de la position après la vente partielle
        remaining_position = position_size * 0.4  # 40% restant après la première vente
        self.portfolio.positions['BTCUSDT'].size = remaining_position
        
        # Vente plus forte (90% de la position restante)
        action = np.array([-0.9, 0.0])
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['asset'], 'BTCUSDT')
        
        # Vérifier que 90% de la position restante est vendue
        # 90% de 0.4 = 0.36, donc on vend -0.36
        expected_units = -0.36
        self.assertAlmostEqual(orders[0]['units'], expected_units, delta=0.001)
        
        # Mettre à jour la position pour le prochain test
        remaining_position = 0.04  # 4% restant après la vente
        self.portfolio.positions['BTCUSDT'].size = remaining_position
        
        # Vente complète (action = -1.0)
        action = np.array([-1.0, 0.0])
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['asset'], 'BTCUSDT')
        
        # Vérifier que la totalité de la position restante est vendue
        self.assertAlmostEqual(orders[0]['units'], -remaining_position, delta=0.001)
        
        # Test avec une action entre -0.5 et 0.5 (ne doit rien faire)
        action = np.array([-0.4, 0.0])
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        self.assertEqual(len(orders), 0)

    def test_hold_action(self):
        """Test d'une action neutre (hold)."""
        action = np.array([0.09, -0.1])  # En dessous du seuil de 0.1
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        self.assertEqual(len(orders), 0)

    def test_multiple_positions(self):
        """Test avec plusieurs positions simultanées."""
        # Activer plusieurs positions
        action = np.array([0.8, 0.9])
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        
        self.assertEqual(len(orders), 2)  # Doit créer 2 ordres
        self.assertEqual({o['asset'] for o in orders}, {'BTCUSDT', 'ETHUSDT'})

    def test_insufficient_funds(self):
        """Test avec fonds insuffisants."""
        # Créer un portefeuille avec très peu de fonds
        self.portfolio = PortfolioManager({
            'environment': {
                'initial_balance': 1.0,  # Pas assez pour un achat
                'assets': self.assets
            },
            'trading_rules': {
                'commission_pct': 0.001, 
                'min_trade_size': 0.0001,
                'min_notional_value': 10.0
            },
            'risk_management': {
                'capital_tiers': [{
                    'name': 'test_tier_low',
                    'min_capital': 0,
                    'max_position_size_pct': 0.5,
                    'risk_per_trade_pct': 1.0,
                    'max_drawdown_pct': 5.0,
                    'leverage': 1.0
                }],
                'position_sizing': {
                    'concentration_limits': {
                        'max_single_asset': 0.5
                    }
                }
            }
        })
        
        # Tenter d'acheter avec des fonds insuffisants
        action = np.array([0.9, 0.0])
        
        # Vérifier qu'aucun ordre n'est généré (car pas assez de fonds)
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        self.assertEqual(len(orders), 0)

    def test_invalid_action_dimensions(self):
        """Test avec une action de dimension incorrecte."""
        action = np.array([0.5])  # Pas assez de dimensions
        with self.assertRaises(ValueError):
            self.translator.translate_action(action, self.portfolio, self.current_prices)

    def test_missing_price(self):
        """Test avec un prix manquant pour un actif."""
        action = np.array([0.5, 0.5])
        with self.assertRaises(KeyError):
            self.translator.translate_action(action, self.portfolio, {'BTCUSDT': 10000})  # ETH manquant

    def test_capital_tier_upgrade(self):
        """Test le calcul de la taille de position avec différents paliers."""
        # Créer un palier personnalisé avec max_position_size_pct à 70%
        custom_tier = {
            'name': 'test_tier',
            'min_capital': 0,
            'max_position_size_pct': 0.7,  # 70% du capital
            'risk_per_trade_pct': 1.0,
            'max_drawdown_pct': 5.0,
            'leverage': 1.0
        }
        
        # Sauvegarder la configuration originale pour la restauration après le test
        original_tiers = self.portfolio.config['risk_management']['capital_tiers']
        
        try:
            # Mettre à jour la configuration avec le nouveau palier
            self.portfolio.config['risk_management']['capital_tiers'] = [custom_tier]
            
            # Action > 0.5 déclenche un achat
            action = np.array([0.6, 0.0])
            orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
            
            # Vérifier qu'un ordre a été créé
            self.assertEqual(len(orders), 1)
            
            # Vérifier que la taille de position est limitée à 70% du capital
            # (10000 * 0.7) / 10000 = 0.7 BTC
            expected_units = 0.7
            self.assertAlmostEqual(orders[0]['units'], expected_units, delta=0.01)
            
        finally:
            # Restaurer la configuration originale
            self.portfolio.config['risk_management']['capital_tiers'] = original_tiers

if __name__ == '__main__':
    unittest.main()
