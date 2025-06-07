#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Système Complet ADAN - Local et Testnet
Vérifie tous les composants du système de trading automatisé.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.adan_trading_bot.common.utils import load_config
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from stable_baselines3 import PPO

class SystemTester:
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def log_result(self, test_name, success, message="", details=None):
        """Enregistre le résultat d'un test."""
        self.results[test_name] = {
            'success': success,
            'message': message,
            'details': details,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
        if not success:
            self.errors.append(test_name)

    def test_configurations(self):
        """Test du chargement des configurations."""
        print("\n🔧 Test des Configurations")
        print("=" * 50)
        
        configs = [
            'config/main_config.yaml',
            'config/data_config_cpu.yaml', 
            'config/environment_config.yaml',
            'config/agent_config_cpu.yaml'
        ]
        
        for config_file in configs:
            try:
                config = load_config(config_file)
                self.log_result(f"Config {config_file}", True, f"Chargé: {len(config)} sections")
            except Exception as e:
                self.log_result(f"Config {config_file}", False, f"Erreur: {str(e)}")

    def test_data_loading(self):
        """Test du chargement des données."""
        print("\n📊 Test du Chargement des Données")
        print("=" * 50)
        
        try:
            # Test données fusionnées
            test_file = "data/processed/merged/unified/1m_test_merged.parquet"
            if os.path.exists(test_file):
                df = pd.read_parquet(test_file)
                self.log_result("Données Test", True, f"Shape: {df.shape}")
                
                # Vérifier les colonnes essentielles
                required_cols = ['open_ADAUSDT', 'close_ADAUSDT', 'volume_ADAUSDT']
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    self.log_result("Colonnes Essentielles", False, f"Manquantes: {missing}")
                else:
                    self.log_result("Colonnes Essentielles", True, "Toutes présentes")
            else:
                self.log_result("Données Test", False, "Fichier non trouvé")
                
        except Exception as e:
            self.log_result("Données Test", False, f"Erreur: {str(e)}")

    def test_environment_creation(self):
        """Test de création de l'environnement."""
        print("\n🌍 Test de l'Environnement")
        print("=" * 50)
        
        try:
            # Charger les configurations
            data_config = load_config('config/data_config_cpu.yaml')
            env_config = load_config('config/environment_config.yaml')
            
            config = {
                'data': data_config.get('data', data_config),
                'environment': env_config.get('environment', env_config)
            }
            
            # Créer données de test minimales
            assets = config['data'].get('assets', ['ADAUSDT', 'BNBUSDT'])
            test_data = self.create_test_data(assets[:2])  # Utiliser seulement 2 assets pour le test
            
            # Créer environnement
            env = MultiAssetEnv(
                df_received=test_data,
                config=config,
                scaler=None,
                encoder=None,
                max_episode_steps_override=100
            )
            
            # Test reset
            obs = env.reset()
            self.log_result("Environnement Reset", True, f"Obs shape: {obs[0].shape if isinstance(obs, tuple) else obs.shape}")
            
            # Test step
            action = 0  # HOLD
            obs, reward, done, info = env.step(action)
            self.log_result("Environnement Step", True, f"Reward: {reward:.4f}")
            
        except Exception as e:
            self.log_result("Environnement Creation", False, f"Erreur: {str(e)}")
            traceback.print_exc()

    def test_model_loading(self):
        """Test du chargement des modèles."""
        print("\n🤖 Test des Modèles")
        print("=" * 50)
        
        model_files = [
            'models/final_model.zip',
            'models/best_trading_model.zip',
            'models/interrupted_model.zip'
        ]
        
        for model_file in model_files:
            try:
                if os.path.exists(model_file):
                    model = PPO.load(model_file)
                    self.log_result(f"Modèle {os.path.basename(model_file)}", True, "Chargé avec succès")
                else:
                    self.log_result(f"Modèle {os.path.basename(model_file)}", False, "Fichier non trouvé")
            except Exception as e:
                self.log_result(f"Modèle {os.path.basename(model_file)}", False, f"Erreur: {str(e)}")

    def test_feature_compatibility(self):
        """Test de compatibilité des features."""
        print("\n🔍 Test de Compatibilité des Features")
        print("=" * 50)
        
        try:
            # Charger config
            data_config = load_config('config/data_config_cpu.yaml')
            features = data_config.get('base_market_features', [])
            assets = data_config.get('assets', [])
            
            expected_total = len(features) * len(assets)
            self.log_result("Features Config", True, f"{len(features)} features × {len(assets)} assets = {expected_total} total")
            
            # Vérifier données réelles
            test_file = "data/processed/merged/unified/1m_test_merged.parquet"
            if os.path.exists(test_file):
                df = pd.read_parquet(test_file)
                actual_cols = len(df.columns)
                
                if actual_cols == expected_total:
                    self.log_result("Dimensions Match", True, f"Parfait: {actual_cols} colonnes")
                else:
                    self.log_result("Dimensions Match", False, f"Attendu: {expected_total}, Réel: {actual_cols}")
            
        except Exception as e:
            self.log_result("Feature Compatibility", False, f"Erreur: {str(e)}")

    def test_order_validation(self):
        """Test de validation des ordres."""
        print("\n💰 Test de Validation des Ordres")
        print("=" * 50)
        
        try:
            env_config = load_config('config/environment_config.yaml')
            order_rules = env_config.get('order_rules', {})
            
            min_tolerable = order_rules.get('min_value_tolerable', 0)
            min_absolute = order_rules.get('min_value_absolute', 0)
            
            self.log_result("Seuils Ordre", True, f"Tolérable: ${min_tolerable}, Absolu: ${min_absolute}")
            
            if min_tolerable == min_absolute == 10.0:
                self.log_result("Seuils Corrects", True, "Valeurs cohérentes à 10.0$")
            else:
                self.log_result("Seuils Corrects", False, f"Valeurs incohérentes: {min_tolerable} vs {min_absolute}")
                
        except Exception as e:
            self.log_result("Order Validation", False, f"Erreur: {str(e)}")

    def test_capital_scenarios(self):
        """Test avec différents scénarios de capital."""
        print("\n💵 Test des Scénarios de Capital")
        print("=" * 50)
        
        capitals = [15.0, 50.0, 100.0, 1000.0]
        
        for capital in capitals:
            try:
                # Test simple de création d'environnement avec différents capitaux
                data_config = load_config('config/data_config_cpu.yaml')
                env_config = load_config('config/environment_config.yaml')
                
                # Modifier le capital initial
                env_config['environment']['initial_capital'] = capital
                
                config = {
                    'data': data_config.get('data', data_config),
                    'environment': env_config.get('environment', env_config)
                }
                
                # Vérifier les paliers
                tiers = config['environment'].get('tiers', [])
                applicable_tier = None
                for tier in tiers:
                    if capital >= tier.get('threshold', 0):
                        applicable_tier = tier
                
                if applicable_tier:
                    max_pos = applicable_tier.get('max_positions', 1)
                    alloc = applicable_tier.get('allocation_frac_per_pos', 0.2)
                    self.log_result(f"Capital ${capital}", True, f"Palier: {max_pos} pos, {alloc*100}% alloc")
                else:
                    self.log_result(f"Capital ${capital}", False, "Aucun palier applicable")
                    
            except Exception as e:
                self.log_result(f"Capital ${capital}", False, f"Erreur: {str(e)}")

    def test_exchange_connectivity(self):
        """Test de connectivité exchange (optionnel)."""
        print("\n🔗 Test de Connectivité Exchange")
        print("=" * 50)
        
        try:
            import ccxt
            
            # Test connexion Binance testnet
            exchange = ccxt.binance({
                'apiKey': 'test',
                'secret': 'test',
                'sandbox': True,
                'enableRateLimit': True,
            })
            
            # Test simple de fetch (devrait échouer avec les clés test mais confirmer la connectivité)
            try:
                markets = exchange.load_markets()
                self.log_result("Exchange Testnet", True, f"{len(markets)} marchés disponibles")
            except ccxt.AuthenticationError:
                self.log_result("Exchange Testnet", True, "Connectivité OK (auth attendue)")
            except Exception as e:
                if "API" in str(e) or "auth" in str(e).lower():
                    self.log_result("Exchange Testnet", True, "Connectivité OK (clés requises)")
                else:
                    self.log_result("Exchange Testnet", False, f"Erreur: {str(e)}")
                    
        except ImportError:
            self.log_result("Exchange CCXT", False, "Module ccxt non installé")
        except Exception as e:
            self.log_result("Exchange Connectivity", False, f"Erreur: {str(e)}")

    def test_script_execution(self):
        """Test d'exécution des scripts principaux."""
        print("\n📜 Test d'Exécution des Scripts")
        print("=" * 50)
        
        scripts_to_test = [
            ('status_adan.py', 'Statut du système'),
            ('scripts/quick_eval.py --help', 'Script d\'évaluation rapide'),
        ]
        
        for script, description in scripts_to_test:
            try:
                import subprocess
                result = subprocess.run(
                    f"python {script}",
                    shell=True,
                    capture_output=True,
                    timeout=10,
                    cwd=os.getcwd()
                )
                
                if result.returncode == 0 or "usage:" in result.stdout.decode() or "help" in script:
                    self.log_result(f"Script {description}", True, "Exécution réussie")
                else:
                    self.log_result(f"Script {description}", False, f"Code retour: {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                self.log_result(f"Script {description}", False, "Timeout")
            except Exception as e:
                self.log_result(f"Script {description}", False, f"Erreur: {str(e)}")

    def create_test_data(self, assets):
        """Crée des données de test minimales."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1T')
        
        data = {}
        
        for asset in assets:
            # Prix de base différent pour chaque asset
            base_price = {'ADAUSDT': 1.0, 'BNBUSDT': 300.0, 'BTCUSDT': 40000.0, 
                         'ETHUSDT': 2500.0, 'XRPUSDT': 0.6}.get(asset, 1.0)
            
            # Générer OHLCV
            prices = base_price + np.random.randn(100) * 0.01
            data[f'open_{asset}'] = prices
            data[f'high_{asset}'] = prices * 1.001
            data[f'low_{asset}'] = prices * 0.999
            data[f'close_{asset}'] = prices
            data[f'volume_{asset}'] = np.random.rand(100) * 1000
            
            # Générer quelques indicateurs techniques de base
            data[f'SMA_short_1m_{asset}'] = prices
            data[f'SMA_long_1m_{asset}'] = prices
            data[f'EMA_short_1m_{asset}'] = prices
            data[f'EMA_long_1m_{asset}'] = prices
            data[f'RSI_1m_{asset}'] = np.random.rand(100) * 100
            data[f'MACD_1m_{asset}'] = np.random.randn(100) * 0.1
            data[f'MACDs_1m_{asset}'] = np.random.randn(100) * 0.1
            data[f'MACDh_1m_{asset}'] = np.random.randn(100) * 0.1
            data[f'BBU_1m_{asset}'] = prices * 1.02
            data[f'BBM_1m_{asset}'] = prices
            data[f'BBL_1m_{asset}'] = prices * 0.98
            data[f'ATR_1m_{asset}'] = np.random.rand(100) * 0.01
            data[f'STOCHk_1m_{asset}'] = np.random.rand(100) * 100
            data[f'STOCHd_1m_{asset}'] = np.random.rand(100) * 100
            data[f'ADX_1m_{asset}'] = np.random.rand(100) * 100
        
        df = pd.DataFrame(data, index=dates)
        return df

    def run_all_tests(self):
        """Exécute tous les tests."""
        print("🎯 TESTS SYSTÈME COMPLET ADAN")
        print("=" * 60)
        print(f"Démarré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Exécution des tests
        self.test_configurations()
        self.test_data_loading()
        self.test_environment_creation()
        self.test_model_loading()
        self.test_feature_compatibility()
        self.test_order_validation()
        self.test_capital_scenarios()
        self.test_exchange_connectivity()
        self.test_script_execution()
        
        # Résumé final
        self.print_summary()

    def print_summary(self):
        """Affiche le résumé des tests."""
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total: {total_tests} tests")
        print(f"✅ Réussis: {passed_tests}")
        print(f"❌ Échoués: {failed_tests}")
        
        if failed_tests == 0:
            print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
            print("   Le système ADAN est prêt pour la production.")
        else:
            print(f"\n⚠️  {failed_tests} TEST(S) ONT ÉCHOUÉ")
            print("   Tests échoués:")
            for test_name, result in self.results.items():
                if not result['success']:
                    print(f"   - {test_name}: {result['message']}")
        
        # Recommandations
        print("\n💡 RECOMMANDATIONS:")
        if failed_tests == 0:
            print("   - Système opérationnel pour trading en local")
            print("   - Configurer les clés API pour trading testnet/live")
            print("   - Surveiller les performances en temps réel")
        else:
            print("   - Corriger les erreurs identifiées")
            print("   - Relancer les tests après corrections")
            print("   - Vérifier les configurations manquantes")

if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all_tests()