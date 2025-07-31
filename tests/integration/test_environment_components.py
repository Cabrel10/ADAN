#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.data_processing.observation_validator import ObservationValidator
from adan_trading_bot.environment.action_translator import ActionTranslator
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

class TestEnvironmentComponentsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour les tests d'intégration."""
        # Configuration minimale pour les tests
        cls.config = {
            'environment': {
                'assets': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['5m', '1h'],
                'window_size': 10,
                'max_steps': 1000,
                'initial_balance': 10000.0,
                'commission': 0.001,
                'memory': {
                    'chunk_size': 1000,
                    'max_chunks_in_memory': 5
                }
            },
            'data': {
                'data_dir': 'data/processed',
                'features_per_timeframe': {
                    '5m': ['close', 'volume', 'rsi'],
                    '1h': ['close', 'volume', 'rsi']
                }
            },
            'trading': {
                'commission_pct': 0.001,
                'min_trade_size': 0.0001,
                'min_notional_value': 10.0
            },
            'risk_management': {
                'capital_tiers': [
                    {'min_capital': 0, 'max_positions': 2, 'allocation_per_trade': 0.5}
                ]
            }
        }
        
        # Créer des données factices pour les tests
        cls.create_test_data()
        
    @classmethod
    def create_test_data(cls):
        """Crée des données factices pour les tests."""
        import os
        os.makedirs('data/processed/BTCUSDT', exist_ok=True)
        os.makedirs('data/processed/ETHUSDT', exist_ok=True)
        
        # Créer des données factices pour BTC
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='5min')
        for tf in ['5m', '1h']:
            df = pd.DataFrame({
                'timestamp': dates,
                'open': 10000 + np.random.randn(1000).cumsum(),
                'high': 10100 + np.random.randn(1000).cumsum(),
                'low': 9900 + np.random.randn(1000).cumsum(),
                'close': 10000 + np.random.randn(1000).cumsum(),
                'volume': 100 + np.random.rand(1000) * 50,
                'rsi': 30 + np.random.rand(1000) * 40
            })
            df.to_parquet(f'data/processed/BTCUSDT/{tf}.parquet')
            
            # Mêmes données pour ETH mais avec des prix différents
            eth_df = df.copy()
            eth_df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] / 20
            eth_df.to_parquet(f'data/processed/ETHUSDT/{tf}.parquet')
    
    def setUp(self):
        """Préparation avant chaque test."""
        self.env = MultiAssetChunkedEnv(self.config, {})
        self.validator = ObservationValidator()
        self.action_translator = ActionTranslator(self.config['environment']['assets'])
        
    def test_environment_initialization(self):
        """Vérifie que l'environnement s'initialise correctement."""
        self.assertIsNotNone(self.env)
        self.assertIsNotNone(self.env.portfolio)
        self.assertIsNotNone(self.env.state_builder)
        
    def test_observation_validation(self):
        """Test l'intégration de la validation des observations."""
        obs = self.env.reset()
        is_valid, metrics = self.validator.validate(obs, return_metrics=True)
        self.assertTrue(is_valid)
        self.assertIn('min_value', metrics)
        self.assertIn('max_value', metrics)
        
    def test_action_translation(self):
        """Test l'intégration de la traduction des actions."""
        # Prendre une action d'achat
        action = np.array([0.8, 0.0])
        orders = self.action_translator.translate_action(
            action, 
            self.env.portfolio, 
            self.env._get_current_prices()
        )
        self.assertGreater(len(orders), 0)
        
    def test_full_step(self):
        """Test un pas complet d'interaction avec l'environnement."""
        obs = self.env.reset()
        
        # Valider l'observation initiale
        is_valid, _ = self.validator.validate(obs, return_metrics=True)
        self.assertTrue(is_valid)
        
        # Prendre une action aléatoire
        action = self.env.action_space.sample()
        
        # Exécuter l'action dans l'environnement
        next_obs, reward, done, info = self.env.step(action)
        
        # Vérifier les résultats
        self.assertIsNotNone(next_obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
    def test_risk_management(self):
        """Vérifie l'intégration de la gestion des risques."""
        # Prendre une position maximale
        action = np.array([1.0, 1.0])
        orders = self.action_translator.translate_action(
            action, 
            self.env.portfolio, 
            self.env._get_current_prices()
        )
        
        # Vérifier que le nombre de positions ne dépasse pas la limite
        self.assertLessEqual(len(orders), 2)  # Selon la configuration
        
    def tearDown(self):
        """Nettoyage après chaque test."""
        if hasattr(self, 'env'):
            self.env.close()

if __name__ == '__main__':
    unittest.main()
