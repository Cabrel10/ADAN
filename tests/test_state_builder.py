import unittest
import numpy as np
import pandas as pd

from adan_trading_bot.data_processing.state_builder import StateBuilder

class TestStateBuilder(unittest.TestCase):
    def setUp(self):
        """Configuration de base pour les tests"""
        # Configuration de test basée sur config.yaml
        self.features_config = {
            "5m": [
                "5m_OPEN", "5m_HIGH", "5m_LOW", "5m_CLOSE", "5m_VOLUME", 
                "5m_RSI_14", "5m_STOCHk_14_3_3", "5m_STOCHd_14_3_3", 
                "5m_CCI_20_0.015", "5m_ROC_9", "5m_MFI_14", "5m_EMA_5", 
                "5m_EMA_20", "5m_SUPERTREND_14_2.0", "5m_PSAR_0.02_0.2"
            ],
            "1h": [
                "1h_OPEN", "1h_HIGH", "1h_LOW", "1h_CLOSE", "1h_VOLUME", 
                "1h_RSI_14", "1h_MACD_12_26_9", "1h_MACD_HIST_12_26_9", 
                "1h_CCI_20_0.015", "1h_MFI_14", "1h_EMA_50", "1h_EMA_100", 
                "1h_SMA_200", "1h_ICHIMOKU_9_26_52", "1h_PSAR_0.02_0.2"
            ],
            "4h": [
                "4h_OPEN", "4h_HIGH", "4h_LOW", "4h_CLOSE", "4h_VOLUME", 
                "4h_RSI_14", "4h_MACD_12_26_9", "4h_CCI_20_0.015", 
                "4h_MFI_14", "4h_EMA_50", "4h_SMA_200", 
                "4h_ICHIMOKU_9_26_52", "4h_SUPERTREND_14_3.0", "4h_PSAR_0.02_0.2"
            ]
        }
        
        # Créer des données de test
        self.test_data = {}
        for timeframe in self.features_config.keys():
            # Générer des données de test pour chaque timeframe
            dates = pd.date_range(start="2023-01-01", periods=100, freq=timeframe)
            # Créer un DataFrame avec toutes les colonnes requises
            # Ajouter le suffixe timeframe à chaque colonne
            base_columns = {
                "OPEN": np.random.rand(100) * 100,
                "HIGH": np.random.rand(100) * 100 + 10,
                "LOW": np.random.rand(100) * 100 - 10,
                "CLOSE": np.random.rand(100) * 100,
                "VOLUME": np.random.rand(100) * 1000,
                "RSI_14": np.random.rand(100) * 100,
                "STOCHk_14_3_3": np.random.rand(100) * 100,
                "STOCHd_14_3_3": np.random.rand(100) * 100,
                "CCI_20_0.015": np.random.rand(100) * 100,
                "ROC_9": np.random.rand(100) * 100,
                "MFI_14": np.random.rand(100) * 100,
                "EMA_5": np.random.rand(100) * 100,
                "EMA_20": np.random.rand(100) * 100,
                "EMA_50": np.random.rand(100) * 100,
                "EMA_100": np.random.rand(100) * 100,
                "SMA_200": np.random.rand(100) * 100,
                "MACD_12_26_9": np.random.rand(100) * 100,
                "MACD_HIST_12_26_9": np.random.rand(100) * 100,
                "SUPERTREND_14_2.0": np.random.rand(100) * 100,
                "SUPERTREND_14_3.0": np.random.rand(100) * 100,
                "PSAR_0.02_0.2": np.random.rand(100) * 100,
                "ICHIMOKU_9_26_52": np.random.rand(100) * 100
            }
            
            # Ajouter le suffixe timeframe à chaque colonne
            data = pd.DataFrame({
                f"{timeframe}_{col}": base_columns[col] 
                for col in base_columns
            })
            data.index = dates
            self.test_data[timeframe] = data

    def test_initialization(self):
        """Teste l'initialisation du StateBuilder"""
        # Test avec configuration par défaut
        state_builder = StateBuilder()
        self.assertIsNotNone(state_builder)
        self.assertEqual(len(state_builder.timeframes), 3)
        self.assertEqual(state_builder.base_window_size, 20)
        
        # Test avec configuration personnalisée
        custom_state_builder = StateBuilder(
            features_config=self.features_config,
            window_size=3,
            include_portfolio_state=True
        )
        self.assertEqual(len(custom_state_builder.timeframes), 3)
        self.assertEqual(custom_state_builder.base_window_size, 3)
        self.assertTrue(custom_state_builder.include_portfolio_state)

    def test_observation_shape(self):
        """Teste la forme de l'observation"""
        # Configuration avec portfolio state désactivé
        state_builder = StateBuilder(
            features_config=self.features_config,
            include_portfolio_state=False
        )
        
        # Vérifier la forme de base
        shape = state_builder.get_observation_shape()
        self.assertEqual(len(shape), 3)
        self.assertEqual(shape[0], 3)  # 3 timeframes
        self.assertEqual(shape[1], 20)  # window_size par défaut
        self.assertEqual(shape[2], 15)  # nombre maximum de features

    def test_build_observation(self):
        """Teste la construction d'une observation"""
        state_builder = StateBuilder(
            features_config=self.features_config,
            window_size=3,
            include_portfolio_state=False
        )
        
        # Initialiser les scalers avec les données
        state_builder.fit_scalers(self.test_data)
        
        # Construire une observation multi-canal à partir des données de test
        observation = state_builder.build_multi_channel_observation(
            current_idx=10,
            data=self.test_data
        )
        
        # Vérifier la forme de l'observation
        self.assertEqual(observation.shape, (3, 3, 15))  # 3 timeframes, window_size=3, 15 features
        
        # Vérifier que les valeurs ne sont pas NaN
        self.assertFalse(np.isnan(observation).any())

if __name__ == '__main__':
    unittest.main()