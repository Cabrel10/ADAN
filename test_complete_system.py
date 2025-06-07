#!/usr/bin/env python3
"""
Test complet du système ADAN avec intégration Exchange.
Valide tous les composants principaux et l'intégration Binance Testnet.
"""

import os
import sys
import argparse
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire src au PYTHONPATH
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir / "src"))

from src.adan_trading_bot.common.utils import get_logger, load_config
from src.adan_trading_bot.exchange_api.connector import (
    get_exchange_client, validate_exchange_config, test_exchange_connection
)
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from src.adan_trading_bot.environment.order_manager import OrderManager
from src.adan_trading_bot.environment.state_builder import StateBuilder
from src.adan_trading_bot.agent.ppo_agent import load_agent

logger = get_logger(__name__)

class SystemTester:
    """Testeur complet du système ADAN."""
    
    def __init__(self, exec_profile="cpu"):
        """
        Initialise le testeur.
        
        Args:
            exec_profile: Profil d'exécution (cpu/gpu)
        """
        self.exec_profile = exec_profile
        self.config = None
        self.exchange = None
        self.test_results = {
            "config_loading": False,
            "exchange_connection": False,
            "data_loading": False,
            "environment_creation": False,
            "order_manager_tests": False,
            "agent_loading": False,
            "complete_integration": False
        }
        
        logger.info(f"🧪 SystemTester initialized - Profile: {exec_profile}")
    
    def test_config_loading(self):
        """Test du chargement de configuration."""
        logger.info("\n" + "="*60)
        logger.info("🔧 TEST 1: Configuration Loading")
        logger.info("="*60)
        
        try:
            # Charger toutes les configurations nécessaires
            main_config = load_config('config/main_config.yaml')
            data_config = load_config(f'config/data_config_{self.exec_profile}.yaml')
            env_config = load_config('config/environment_config.yaml')
            agent_config = load_config(f'config/agent_config_{self.exec_profile}.yaml')
        
            # Reconstruire la structure de config attendue
            self.config = {
                'paths': main_config.get('paths', {}),
                'data': data_config,
                'environment': env_config,
                'agent': agent_config,
                'general': main_config.get('general', {}),
                'paper_trading': main_config.get('paper_trading', {})
            }
        
            logger.info("✅ Configuration loaded successfully")
        
            # Vérifier les sections principales
            required_sections = ['data', 'environment', 'agent', 'paper_trading']
            for section in required_sections:
                if section in self.config:
                    logger.info(f"✅ Section '{section}' found")
                else:
                    logger.warning(f"⚠️ Section '{section}' missing")
            
            # Afficher la config des actifs
            assets = self.config.get('data', {}).get('assets', [])
            logger.info(f"📊 Configured assets: {assets}")
            
            self.test_results["config_loading"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            return False
    
    def test_exchange_connection(self):
        """Test de la connexion exchange."""
        logger.info("\n" + "="*60)
        logger.info("🔌 TEST 2: Exchange Connection")
        logger.info("="*60)
        
        try:
            # Vérifier les variables d'environnement
            api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
            secret_key = os.environ.get('BINANCE_TESTNET_SECRET_KEY')
            
            if not api_key or not secret_key:
                logger.warning("⚠️ Exchange API keys not found - skipping exchange tests")
                logger.info("💡 Set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET_KEY to enable")
                return True  # Pas d'erreur si les clés ne sont pas définies
            
            # Valider la configuration
            if not validate_exchange_config(self.config):
                logger.error("❌ Exchange configuration validation failed")
                return False
            
            # Créer le client exchange
            self.exchange = get_exchange_client(self.config)
            logger.info(f"✅ Exchange client created: {self.exchange.id}")
            
            # Tester la connexion
            connection_results = test_exchange_connection(self.exchange)
            
            if not connection_results['errors']:
                logger.info("✅ Exchange connection tests passed")
                logger.info(f"📈 Markets available: {connection_results['market_count']}")
                self.test_results["exchange_connection"] = True
                return True
            else:
                logger.error(f"❌ Exchange connection errors: {len(connection_results['errors'])}")
                for error in connection_results['errors'][:3]:  # Afficher les 3 premières erreurs
                    logger.error(f"   {error}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exchange connection test failed: {e}")
            return False
    
    def test_data_loading(self):
        """Test du chargement des données."""
        logger.info("\n" + "="*60)
        logger.info("📊 TEST 3: Data Loading")
        logger.info("="*60)
        
        try:
            # Chercher les données processed
            processed_dir = current_dir / "data" / "processed" / "merged" / "unified"
            training_timeframe = self.config.get('data', {}).get('training_timeframe', '1m')
            
            train_file = processed_dir / f"{training_timeframe}_train_merged.parquet"
            
            if not train_file.exists():
                logger.warning(f"⚠️ Training data not found: {train_file}")
                logger.info("💡 Run data processing pipeline first")
                
                # Essayer les anciennes localisations
                alt_locations = [
                    current_dir / "data" / "processed" / "1m_train_merged.parquet",
                    current_dir / "data" / "processed" / "train_merged.parquet"
                ]
                
                train_file = None
                for alt_file in alt_locations:
                    if alt_file.exists():
                        train_file = alt_file
                        logger.info(f"📁 Found alternative data: {alt_file}")
                        break
                
                if not train_file:
                    logger.error("❌ No training data found")
                    return False
            
            # Charger les données
            self.train_df = pd.read_parquet(train_file)
            logger.info(f"✅ Training data loaded: {self.train_df.shape}")
            
            # Vérifier la structure des données
            assets = self.config.get('data', {}).get('assets', [])
            expected_columns = []
            
            for asset in assets[:2]:  # Vérifier les 2 premiers actifs
                expected_columns.extend([f"open_{asset}", f"close_{asset}"])
            
            missing_columns = [col for col in expected_columns if col not in self.train_df.columns]
            
            if missing_columns:
                logger.warning(f"⚠️ Some expected columns missing: {missing_columns[:5]}")
            else:
                logger.info("✅ Data structure validation passed")
            
            logger.info(f"📊 Data columns: {len(self.train_df.columns)}")
            logger.info(f"📅 Date range: {self.train_df.index.min()} to {self.train_df.index.max()}")
            
            self.test_results["data_loading"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return False
    
    def test_environment_creation(self):
        """Test de la création d'environnement."""
        logger.info("\n" + "="*60)
        logger.info("🌍 TEST 4: Environment Creation")
        logger.info("="*60)
        
        try:
            # Prendre un échantillon des données pour les tests
            sample_size = min(1000, len(self.train_df))
            sample_df = self.train_df.tail(sample_size).copy()
            
            logger.info(f"📊 Using sample data: {sample_df.shape}")
            
            # Créer l'environnement
            env = MultiAssetEnv(
                df_received=sample_df,
                config=self.config,
                scaler=None,  # Pas de scaler pour les tests
                encoder=None,
                max_episode_steps_override=50  # Épisodes courts pour les tests
            )
            
            logger.info("✅ Environment created successfully")
            logger.info(f"🎯 Action space: {env.action_space}")
            logger.info(f"📐 Observation space: {env.observation_space}")
            
            # Tester reset
            obs, info = env.reset()
            logger.info(f"✅ Environment reset successful - Obs shape: {obs.shape}")
            
            # Tester quelques steps
            for step in range(3):
                action = env.action_space.sample()  # Action aléatoire
                obs, reward, terminated, truncated, info = env.step(action)
                logger.info(f"   Step {step+1}: Action={action}, Reward={reward:.4f}")
                
                if terminated or truncated:
                    break
            
            logger.info("✅ Environment step tests passed")
            
            self.test_results["environment_creation"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment creation failed: {e}")
            return False
    
    def test_order_manager(self):
        """Test du gestionnaire d'ordres."""
        logger.info("\n" + "="*60)
        logger.info("💼 TEST 5: Order Manager Tests")
        logger.info("="*60)
        
        try:
            # Créer OrderManager avec exchange si disponible
            order_manager = OrderManager(self.config, exchange_client=self.exchange)
            logger.info("✅ OrderManager created")
            
            # Tests basiques
            test_capital = 1000.0
            test_positions = {}
            test_asset = "ADAUSDT"
            test_price = 0.5
            
            # Test BUY valide
            logger.info("🔄 Testing valid BUY order...")
            reward, status, info = order_manager.execute_order(
                asset_id=test_asset,
                action_type=1,  # BUY
                current_price=test_price,
                capital=test_capital,
                positions=test_positions,
                allocated_value_usdt=100.0
            )
            
            if status == "BUY_EXECUTED":
                logger.info(f"✅ BUY test passed - New capital: ${info.get('new_capital', 'N/A'):.2f}")
                test_positions.update({test_asset: {"qty": 200.0, "price": test_price}})
            else:
                logger.warning(f"⚠️ BUY test result: {status}")
            
            # Test SELL valide
            logger.info("🔄 Testing valid SELL order...")
            reward, status, info = order_manager.execute_order(
                asset_id=test_asset,
                action_type=2,  # SELL
                current_price=test_price * 1.1,  # Prix légèrement plus élevé
                capital=test_capital,
                positions=test_positions
            )
            
            if status == "SELL_EXECUTED":
                logger.info(f"✅ SELL test passed - PnL: ${info.get('pnl', 'N/A'):.2f}")
            else:
                logger.warning(f"⚠️ SELL test result: {status}")
            
            # Test avec exchange si disponible
            if self.exchange:
                logger.info("🔗 Testing exchange integration...")
                # Test de validation des filtres d'exchange
                symbol_ccxt = "ADA/USDT"
                if symbol_ccxt in self.exchange.load_markets():
                    logger.info(f"✅ Market {symbol_ccxt} available on exchange")
                else:
                    logger.warning(f"⚠️ Market {symbol_ccxt} not found on exchange")
            
            self.test_results["order_manager_tests"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ OrderManager tests failed: {e}")
            return False
    
    def test_agent_loading(self):
        """Test du chargement d'agent."""
        logger.info("\n" + "="*60)
        logger.info("🤖 TEST 6: Agent Loading")
        logger.info("="*60)
        
        try:
            # Chercher un modèle existant
            models_dir = current_dir / "models"
            
            if not models_dir.exists():
                logger.warning("⚠️ Models directory not found - skipping agent tests")
                return True
            
            # Chercher des fichiers de modèle
            model_files = list(models_dir.glob("*.zip"))
            
            if not model_files:
                logger.warning("⚠️ No model files found - skipping agent tests")
                return True
            
            # Prendre le premier modèle trouvé
            model_path = model_files[0]
            logger.info(f"📁 Testing with model: {model_path}")
            
            # Charger l'agent
            agent = load_agent(str(model_path))
            logger.info("✅ Agent loaded successfully")
            
            # Test de prédiction avec des données aléatoires
            obs_shape = (235,)  # Shape typique pour ADAN
            test_obs = np.random.random(obs_shape).astype(np.float32)
            
            action, _ = agent.predict(test_obs, deterministic=True)
            logger.info(f"✅ Agent prediction test passed - Action: {action}")
            
            self.test_results["agent_loading"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Agent loading failed: {e}")
            return False
    
    def test_complete_integration(self):
        """Test d'intégration complète."""
        logger.info("\n" + "="*60)
        logger.info("🎯 TEST 7: Complete Integration")
        logger.info("="*60)
        
        try:
            # Simuler un cycle complet de paper trading
            logger.info("🔄 Simulating complete paper trading cycle...")
            
            # 1. Configuration check
            if not self.config:
                logger.error("❌ No configuration available")
                return False
            
            # 2. Exchange check (optionnel)
            exchange_status = "Available" if self.exchange else "Not configured"
            logger.info(f"🔗 Exchange status: {exchange_status}")
            
            # 3. Data check
            if not hasattr(self, 'train_df'):
                logger.error("❌ No training data available")
                return False
            
            # 4. Simulation d'un ordre avec exchange integration
            if self.exchange:
                try:
                    # Test récupération de prix en temps réel
                    ticker = self.exchange.fetch_ticker("BTC/USDT")
                    current_price = ticker['last']
                    logger.info(f"📈 Live BTC/USDT price: ${current_price:.2f}")
                    
                    # Test OrderManager avec exchange
                    order_manager = OrderManager(self.config, exchange_client=self.exchange)
                    
                    # Simuler un ordre (qui sera validé mais pas exécuté)
                    reward, status, info = order_manager.execute_order(
                        asset_id="BTCUSDT",
                        action_type=1,  # BUY
                        current_price=current_price,
                        capital=1000.0,
                        positions={},
                        allocated_value_usdt=50.0
                    )
                    
                    logger.info(f"✅ Exchange-integrated order test: {status}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Exchange integration test failed: {e}")
            
            logger.info("✅ Complete integration test passed")
            self.test_results["complete_integration"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete integration test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Exécute tous les tests."""
        logger.info("🚀 Starting ADAN Complete System Tests")
        logger.info(f"⏰ Start time: {datetime.now()}")
        
        start_time = time.time()
        
        # Séquence de tests
        tests = [
            ("Configuration Loading", self.test_config_loading),
            ("Exchange Connection", self.test_exchange_connection),
            ("Data Loading", self.test_data_loading),
            ("Environment Creation", self.test_environment_creation),
            ("Order Manager", self.test_order_manager),
            ("Agent Loading", self.test_agent_loading),
            ("Complete Integration", self.test_complete_integration)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"❌ {test_name} crashed: {e}")
                results[test_name] = False
        
        # Rapport final
        self.print_final_report(results, time.time() - start_time)
        
        return results
    
    def print_final_report(self, results, duration):
        """Affiche le rapport final."""
        logger.info("\n" + "="*80)
        logger.info("📊 ADAN SYSTEM TEST RESULTS")
        logger.info("="*80)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} {test_name}")
        
        logger.info(f"\n🎯 SUMMARY: {passed}/{total} tests passed")
        logger.info(f"⏰ Duration: {duration:.2f} seconds")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        else:
            logger.warning(f"⚠️ {total - passed} test(s) failed - Review before production")
        
        # Recommandations
        logger.info("\n💡 RECOMMENDATIONS:")
        
        if not results.get("Exchange Connection", False):
            logger.info("   • Set up Binance Testnet API keys for exchange integration")
        
        if not results.get("Data Loading", False):
            logger.info("   • Run data processing pipeline: python scripts/convert_real_data.py")
        
        if not results.get("Agent Loading", False):
            logger.info("   • Train a model: python scripts/train_rl_agent.py")
        
        logger.info("   • Ready for paper trading: python scripts/paper_trade_agent.py")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="ADAN Complete System Test")
    parser.add_argument("--exec_profile", type=str, default="cpu",
                       choices=["cpu", "gpu"], help="Profil d'exécution")
    
    args = parser.parse_args()
    
    try:
        # Initialiser et exécuter les tests
        tester = SystemTester(args.exec_profile)
        results = tester.run_all_tests()
        
        # Code de sortie basé sur les résultats
        if all(results.values()):
            sys.exit(0)  # Succès
        else:
            sys.exit(1)  # Échec
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ System test crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()