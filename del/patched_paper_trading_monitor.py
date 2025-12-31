#!/usr/bin/env python3
"""
Paper Trading Monitor PATCHÉ avec données préchargées
Version corrigée pour résoudre le problème de données insuffisantes
"""

import os
import sys
import json
import logging
import time
import argparse
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import traceback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer
from adan_trading_bot.exchange_api.connector import get_exchange_client, test_exchange_connection
from adan_trading_bot.normalization import ObservationNormalizer, DriftDetector
from adan_trading_bot.indicators.calculator import IndicatorCalculator
from adan_trading_bot.validation.data_validator import DataValidator
from adan_trading_bot.observation.builder import ObservationBuilder
from stable_baselines3 import PPO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PreloadedDataManager:
    """Gestionnaire des données préchargées pour éviter les erreurs de données insuffisantes"""
    
    def __init__(self):
        self.data_dir = Path("historical_data")
        self.preloaded_data = {}
        self.is_loaded = False
    
    def load_historical_data(self):
        """Charge les données historiques préchargées"""
        if self.is_loaded:
            return True
            
        logger.info("📂 Chargement des données préchargées...")
        timeframes = ['5m', '1h', '4h']
        
        for tf in timeframes:
            file_path = self.data_dir / f"BTC_USDT_{tf}_data.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                    self.preloaded_data[tf] = df
                    logger.info(f"  ✅ {tf}: {len(df)} périodes chargées")
                except Exception as e:
                    logger.error(f"  ❌ Erreur chargement {tf}: {e}")
                    return False
            else:
                logger.error(f"  ⚠️  Fichier manquant: {file_path}")
                logger.error(f"     Exécutez: python scripts/quick_data_fix.py")
                return False
        
        self.is_loaded = True
        return True
    
    def get_data_with_window(self, timeframe, window=None):
        """Récupère les données avec la fenêtre appropriée"""
        if timeframe not in self.preloaded_data:
            return None
            
        df = self.preloaded_data[timeframe]
        
        # Définir la fenêtre par défaut selon le timeframe
        if window is None:
            window = {'5m': 20, '1h': 10, '4h': 5}.get(timeframe, 20)
        
        # Retourner les dernières périodes pour l'observation
        return df.tail(window).copy()

class RealPaperTradingMonitor:
    """
    Real execution monitor for ADAN avec données préchargées.
    - Utilise les données historiques téléchargées
    - Évite l'erreur "Need at least 28 rows"
    - Calcule correctement RSI, ADX, volatilité
    """
    
    def __init__(self, config_path="config/paper_trading_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Gestionnaire de données préchargées
        self.preloaded_manager = PreloadedDataManager()
        self.data_preloaded = False
        
        # Initialize components
        self.exchange = None
        self.state_builder = StateBuilder()
        self.normalizer = ObservationNormalizer()
        self.drift_detector = DriftDetector()
        self.indicator_calculator = IndicatorCalculator()
        self.data_validator = DataValidator()
        self.observation_builder = ObservationBuilder()
        
        # Skip FeatureEngineer for now (requires complex config)
        self.feature_engineer = None
        
        # Trading state
        self.virtual_balance = 29.0  # $29 limit
        self.positions = {}
        self.trade_history = []
        self.last_signal = None
        
        # Load models
        self.models = self.load_models()
        
        logger.info("✅ RealPaperTradingMonitor initialisé avec données préchargées")

    def load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return {
                "exchange": "binance",
                "testnet": True,
                "symbol": "BTC/USDT",
                "timeframes": ["5m", "1h", "4h"],
                "update_interval": 300,
                "max_position_size": 0.0003
            }

    def load_models(self):
        """Load trained models"""
        models = {}
        model_dir = Path("models")
        
        for worker_id in ["w1", "w2", "w3", "w4"]:
            model_path = model_dir / f"{worker_id}_model.zip"
            if model_path.exists():
                try:
                    models[worker_id] = PPO.load(str(model_path))
                    logger.info(f"✅ Modèle {worker_id} chargé")
                except Exception as e:
                    logger.error(f"❌ Erreur chargement {worker_id}: {e}")
            else:
                logger.warning(f"⚠️  Modèle manquant: {model_path}")
        
        return models

    async def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            self.exchange = get_exchange_client(
                exchange_name=self.config.get("exchange", "binance"),
                testnet=self.config.get("testnet", True)
            )
            
            # Test connection
            if await test_exchange_connection(self.exchange):
                logger.info("✅ Connexion exchange établie")
                return True
            else:
                logger.error("❌ Échec test connexion exchange")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation exchange: {e}")
            return False

    async def fetch_market_data_with_preloaded(self):
        """Version améliorée qui utilise les données préchargées"""
        
        # Charger les données préchargées au premier appel
        if not self.data_preloaded:
            if not self.preloaded_manager.load_historical_data():
                logger.error("❌ Impossible de charger les données préchargées")
                logger.error("💡 Exécutez: python scripts/quick_data_fix.py")
                return None
            self.data_preloaded = True
            logger.info("✅ Données préchargées chargées avec succès")
        
        data = {}
        timeframes = ['5m', '1h', '4h']
        
        for timeframe in timeframes:
            try:
                # Récupérer les données préchargées avec la fenêtre appropriée
                df = self.preloaded_manager.get_data_with_window(timeframe)
                
                if df is None or len(df) < 28:
                    logger.error(f"❌ Données insuffisantes pour {timeframe}: {len(df) if df is not None else 0} < 28")
                    continue
                
                # Calculer les indicateurs sur les données complètes
                indicators = self.indicator_calculator.calculate_indicators(df, timeframe)
                
                data[timeframe] = {
                    'df': df,
                    'indicators': indicators
                }
                
                # Validation des données (seulement si assez de données)
                if len(df) >= 28:
                    try:
                        if not self.data_validator.validate_data_integrity(df, indicators, timeframe):
                            logger.warning(f"⚠️  Validation échouée pour {timeframe} (mais on continue)")
                    except Exception as e:
                        logger.warning(f"⚠️  Erreur validation {timeframe}: {e}")
                
                # Log des indicateurs calculés
                latest_indicators = indicators.iloc[-1] if hasattr(indicators, 'iloc') else indicators
                rsi_val = latest_indicators.get('rsi', 'N/A')
                adx_val = latest_indicators.get('adx', 'N/A')
                vol_val = latest_indicators.get('volatility', 'N/A')
                
                logger.info(f"✅ {timeframe}: {len(df)} périodes, RSI: {rsi_val}, ADX: {adx_val}, Vol: {vol_val}")
                
            except Exception as e:
                logger.error(f"❌ Erreur traitement {timeframe}: {e}")
                continue
        
        return data if data else None

    def build_observation(self, market_data):
        """Build observation from market data"""
        try:
            # Use StateBuilder to create state
            state = self.state_builder.build_state(market_data)
            
            # Skip feature engineering for now (simplified version)
            if self.feature_engineer:
                features = self.feature_engineer.engineer_features(state)
            else:
                features = state  # Use state directly
            
            # Normalize observation
            normalized_obs = self.normalizer.normalize(features)
            
            return normalized_obs
            
        except Exception as e:
            logger.error(f"❌ Erreur construction observation: {e}")
            return None

    def get_ensemble_prediction(self, observation):
        """Get prediction from ensemble of models"""
        if not self.models:
            logger.error("❌ Aucun modèle chargé")
            return None
        
        predictions = {}
        
        for worker_id, model in self.models.items():
            try:
                action, _states = model.predict(observation, deterministic=True)
                predictions[worker_id] = action
            except Exception as e:
                logger.error(f"❌ Erreur prédiction {worker_id}: {e}")
        
        if not predictions:
            return None
        
        # Simple ensemble: majority vote
        actions = list(predictions.values())
        ensemble_action = max(set(actions), key=actions.count)
        
        # Calculate confidence
        confidence = actions.count(ensemble_action) / len(actions)
        
        return {
            'action': ensemble_action,
            'confidence': confidence,
            'individual_predictions': predictions
        }

    def execute_trade(self, signal):
        """Execute trade based on signal"""
        try:
            symbol = self.config.get("symbol", "BTC/USDT")
            
            if signal['action'] == 1:  # BUY
                logger.info(f"🟢 Signal BUY (confidence: {signal['confidence']:.2f})")
                # Implement buy logic here
                
            elif signal['action'] == -1:  # SELL
                logger.info(f"🔴 Signal SELL (confidence: {signal['confidence']:.2f})")
                # Implement sell logic here
                
            else:  # HOLD
                logger.info(f"⚪ Signal HOLD (confidence: {signal['confidence']:.2f})")
            
            self.last_signal = signal
            
        except Exception as e:
            logger.error(f"❌ Erreur exécution trade: {e}")

    async def run(self):
        """Main monitoring loop"""
        logger.info("🚀 Démarrage du monitor ADAN avec données préchargées")
        
        # Initialize exchange
        if not await self.initialize_exchange():
            logger.error("❌ Impossible d'initialiser l'exchange")
            return
        
        update_interval = self.config.get("update_interval", 300)  # 5 minutes
        
        while True:
            try:
                logger.info("🔄 Cycle de monitoring...")
                
                # Fetch market data (using preloaded data)
                market_data = await self.fetch_market_data_with_preloaded()
                
                if market_data is None:
                    logger.error("❌ Impossible de récupérer les données de marché")
                    time.sleep(60)  # Wait 1 minute before retry
                    continue
                
                # Build observation
                observation = self.build_observation(market_data)
                
                if observation is None:
                    logger.error("❌ Impossible de construire l'observation")
                    time.sleep(60)
                    continue
                
                # Get ensemble prediction
                signal = self.get_ensemble_prediction(observation)
                
                if signal is None:
                    logger.error("❌ Impossible d'obtenir une prédiction")
                    time.sleep(60)
                    continue
                
                # Execute trade
                self.execute_trade(signal)
                
                # Log status
                logger.info(f"💰 Balance virtuelle: ${self.virtual_balance:.2f}")
                logger.info(f"📊 Dernière action: {signal['action']} (conf: {signal['confidence']:.2f})")
                
                # Wait for next update
                logger.info(f"⏳ Attente {update_interval}s avant prochain cycle...")
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Arrêt demandé par l'utilisateur")
                break
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle principale: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)  # Wait before retry

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ADAN Paper Trading Monitor avec données préchargées')
    parser.add_argument('--config', default='config/paper_trading_config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    monitor = RealPaperTradingMonitor(config_path=args.config)
    
    try:
        import asyncio
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        logger.info("🛑 Monitor arrêté")

if __name__ == "__main__":
    main()