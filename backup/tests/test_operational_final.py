#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Opérationnel Final ADAN
Validation complète du système de trading automatisé pour production.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import subprocess
import time
from pathlib import Path

# Ajouter le répertoire parent au path
from adan_trading_bot.common.utils import load_config
from adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from stable_baselines3 import PPO

class OperationalTester:
    def __init__(self):
        self.results = {}
        self.critical_errors = []
        self.warnings = []
        self.start_time = datetime.now()

    def log_test(self, name, success, details="", critical=False):
        """Enregistre un résultat de test."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.results[name] = {"success": success, "details": details, "critical": critical}

        if not success and critical:
            self.critical_errors.append(name)
            status = "🚨 CRITICAL FAIL"

        print(f"{status}: {name}")
        if details:
            print(f"    {details}")

        return success

    def test_system_requirements(self):
        """Test des prérequis système."""
        print("\n🔧 TESTS DES PRÉREQUIS SYSTÈME")
        print("=" * 60)

        # Test Python version
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 8:
            self.log_test("Python Version", True, f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self.log_test("Python Version", False, f"Python {python_version.major}.{python_version.minor} (requis: 3.8+)", critical=True)

        # Test packages critiques
        critical_packages = ['pandas', 'numpy', 'yaml', 'stable_baselines3']
        for package in critical_packages:
            try:
                __import__(package)
                self.log_test(f"Package {package}", True)
            except ImportError:
                self.log_test(f"Package {package}", False, "Module manquant", critical=True)

        # Test structure des répertoires
        critical_dirs = ['config', 'data/processed/merged/unified', 'models', 'src']
        for dir_path in critical_dirs:
            if os.path.exists(dir_path):
                self.log_test(f"Répertoire {dir_path}", True)
            else:
                self.log_test(f"Répertoire {dir_path}", False, "Répertoire manquant", critical=True)

    def test_configuration_integrity(self):
        """Test de l'intégrité des configurations."""
        print("\n⚙️ TESTS DE CONFIGURATION")
        print("=" * 60)

        configs = [
            ('config/main_config.yaml', False),
            ('config/data_config_cpu.yaml', True),
            ('config/environment_config.yaml', True),
            ('config/agent_config_cpu.yaml', False)
        ]

        for config_path, is_critical in configs:
            try:
                config = load_config(config_path)

                # Validations spécifiques
                if 'data_config_cpu.yaml' in config_path:
                    assets = config.get('assets', [])
                    features = config.get('base_market_features', [])
                    expected_features = len(features) * len(assets)

                    self.log_test(f"Config {config_path}", True,
                                f"{len(assets)} assets, {len(features)} features, {expected_features} total", is_critical)

                    if expected_features != 235:
                        self.log_test("Feature Count", False, f"Expected 235, got {expected_features}", is_critical)
                    else:
                        self.log_test("Feature Count", True, "235 features (compatible avec modèles)")

                elif 'environment_config.yaml' in config_path:
                    order_rules = config.get('order_rules', {})
                    min_abs = order_rules.get('min_value_absolute', 0)
                    min_tol = order_rules.get('min_value_tolerable', 0)

                    self.log_test(f"Config {config_path}", True, f"Min order: ${min_abs}")

                    if min_abs == min_tol == 10.0:
                        self.log_test("Order Thresholds", True, "Cohérents à 10.0$")
                    else:
                        self.log_test("Order Thresholds", False, f"Incohérents: {min_abs} vs {min_tol}")

                else:
                    self.log_test(f"Config {config_path}", True, f"{len(config)} sections")

            except Exception as e:
                self.log_test(f"Config {config_path}", False, f"Erreur: {str(e)}", is_critical)

    def test_data_availability(self):
        """Test de disponibilité des données."""
        print("\n📊 TESTS DES DONNÉES")
        print("=" * 60)

        data_files = [
            ('data/processed/merged/unified/1m_train_merged.parquet', True),
            ('data/processed/merged/unified/1m_val_merged.parquet', False),
            ('data/processed/merged/unified/1m_test_merged.parquet', True)
        ]

        for file_path, is_critical in data_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    self.log_test(f"Data {os.path.basename(file_path)}", True,
                                f"Shape: {df.shape}", is_critical)

                    # Test colonnes critiques pour fichier de test
                    if 'test' in file_path:
                        required_cols = ['open_ADAUSDT', 'close_ADAUSDT', 'SMA_short_1m_ADAUSDT']
                        missing = [col for col in required_cols if col not in df.columns]
                        if missing:
                            self.log_test("Test Data Columns", False, f"Manquantes: {missing[:3]}...", True)
                        else:
                            self.log_test("Test Data Columns", True, "Colonnes essentielles présentes")

                except Exception as e:
                    self.log_test(f"Data {os.path.basename(file_path)}", False, f"Lecture échouée: {str(e)}", is_critical)
            else:
                self.log_test(f"Data {os.path.basename(file_path)}", False, "Fichier manquant", is_critical)

    def test_model_functionality(self):
        """Test de fonctionnalité des modèles."""
        print("\n🤖 TESTS DES MODÈLES")
        print("=" * 60)

        model_path = 'models/final_model.zip'

        if os.path.exists(model_path):
            try:
                model = PPO.load(model_path)
                self.log_test("Model Loading", True, f"Modèle chargé: {os.path.basename(model_path)}")

                # Test de prédiction avec données synthétiques
                test_obs = {
                    'image_features': np.random.randn(1, 1, 20, 235).astype(np.float32),
                    'vector_features': np.random.randn(1, 6).astype(np.float32)
                }

                try:
                    action, _ = model.predict(test_obs, deterministic=True)
                    self.log_test("Model Prediction", True, f"Action prédite: {action[0]}")
                except Exception as e:
                    self.log_test("Model Prediction", False, f"Prédiction échouée: {str(e)}", True)

            except Exception as e:
                self.log_test("Model Loading", False, f"Chargement échoué: {str(e)}", True)
        else:
            self.log_test("Model Loading", False, "Modèle principal manquant", True)

    def test_environment_functionality(self):
        """Test de fonctionnalité de l'environnement."""
        print("\n🌍 TESTS DE L'ENVIRONNEMENT")
        print("=" * 60)

        try:
            # Charger configurations
            data_config = load_config('config/data_config_cpu.yaml')
            env_config = load_config('config/environment_config.yaml')

            config = {
                'data': data_config,
                'environment': env_config
            }

            # Charger vraies données de test
            test_file = "data/processed/merged/unified/1m_test_merged.parquet"
            if os.path.exists(test_file):
                df_test = pd.read_parquet(test_file)

                # Créer environnement
                env = MultiAssetEnv(
                    df_received=df_test,
                    config=config,
                    scaler=None,
                    encoder=None,
                    max_episode_steps_override=100
                )

                self.log_test("Environment Creation", True, "Environnement créé avec données réelles")

                # Test reset
                obs = env.reset()
                if isinstance(obs, (tuple, list)):
                    obs_shape = obs[0].shape if hasattr(obs[0], 'shape') else "dict"
                elif isinstance(obs, dict):
                    obs_shape = f"dict with {len(obs)} keys"
                else:
                    obs_shape = obs.shape if hasattr(obs, 'shape') else str(type(obs))

                self.log_test("Environment Reset", True, f"Observation: {obs_shape}")

                # Test plusieurs steps
                successful_steps = 0
                for i in range(10):
                    try:
                        action = 0  # HOLD
                        obs, reward, done, info = env.step(action)
                        successful_steps += 1
                        if done:
                            break
                    except Exception as e:
                        break

                if successful_steps >= 5:
                    self.log_test("Environment Steps", True, f"{successful_steps} steps réussis")
                else:
                    self.log_test("Environment Steps", False, f"Seulement {successful_steps} steps réussis")

            else:
                self.log_test("Environment Creation", False, "Données de test manquantes", True)

        except Exception as e:
            self.log_test("Environment Creation", False, f"Erreur: {str(e)}", True)

    def test_trading_logic(self):
        """Test de la logique de trading."""
        print("\n💰 TESTS DE LOGIQUE DE TRADING")
        print("=" * 60)

        try:
            env_config = load_config('config/environment_config.yaml')

            # Test des seuils d'ordre
            order_rules = env_config.get('order_rules', {})
            min_abs = order_rules.get('min_value_absolute', 0)
            min_tol = order_rules.get('min_value_tolerable', 0)

            self.log_test("Order Thresholds", True, f"Absolu: ${min_abs}, Tolérable: ${min_tol}")

            # Test des paliers de capital
            tiers = env_config.get('tiers', [])
            if tiers:
                test_capitals = [15.0, 100.0, 1000.0, 15000.0]
                for capital in test_capitals:
                    applicable_tier = None
                    for tier in reversed(tiers):  # Du plus élevé au plus bas
                        if capital >= tier.get('threshold', 0):
                            applicable_tier = tier
                            break

                    if applicable_tier:
                        max_pos = applicable_tier.get('max_positions', 1)
                        alloc = applicable_tier.get('allocation_frac_per_pos', 0.2)
                        self.log_test(f"Capital ${capital}", True, f"{max_pos} pos max, {alloc*100}% alloc")
                    else:
                        self.log_test(f"Capital ${capital}", False, "Aucun palier applicable")
            else:
                self.log_test("Capital Tiers", False, "Aucun palier défini")

        except Exception as e:
            self.log_test("Trading Logic", False, f"Erreur: {str(e)}")

    def test_script_execution(self):
        """Test d'exécution des scripts critiques."""
        print("\n📜 TESTS D'EXÉCUTION")
        print("=" * 60)

        # Test script de statut (critique pour monitoring)
        try:
            result = subprocess.run(['python', 'status_adan.py'],
                                  capture_output=True, timeout=30, text=True)
            if result.returncode == 0:
                self.log_test("Status Script", True, "Statut généré avec succès")
            else:
                self.log_test("Status Script", False, f"Code retour: {result.returncode}", True)
        except subprocess.TimeoutExpired:
            self.log_test("Status Script", False, "Timeout", True)
        except Exception as e:
            self.log_test("Status Script", False, f"Erreur: {str(e)}", True)

        # Test script d'évaluation rapide
        try:
            result = subprocess.run(['python', 'scripts/quick_eval.py', '--help'],
                                  capture_output=True, timeout=10, text=True)
            if "usage:" in result.stdout or result.returncode == 0:
                self.log_test("Eval Script", True, "Script d'évaluation accessible")
            else:
                self.log_test("Eval Script", False, "Script d'évaluation inaccessible")
        except Exception as e:
            self.log_test("Eval Script", False, f"Erreur: {str(e)}")

    def test_exchange_readiness(self):
        """Test de préparation pour exchange."""
        print("\n🔗 TESTS DE PRÉPARATION EXCHANGE")
        print("=" * 60)

        try:
            import ccxt
            self.log_test("CCXT Library", True, "Bibliothèque d'exchange installée")

            # Test création exchange (mode sandbox)
            try:
                exchange = ccxt.binance({
                    'sandbox': True,
                    'enableRateLimit': True,
                })
                self.log_test("Exchange Setup", True, "Configuration exchange réussie")

                # Test de connectivité de base (sans clés)
                try:
                    markets = exchange.load_markets()
                    self.log_test("Exchange Connectivity", True, f"{len(markets)} marchés disponibles")
                except ccxt.AuthenticationError:
                    self.log_test("Exchange Connectivity", True, "Connectivité OK (clés API requises)")
                except Exception as e:
                    if "api" in str(e).lower() or "auth" in str(e).lower():
                        self.log_test("Exchange Connectivity", True, "Connectivité OK")
                    else:
                        self.log_test("Exchange Connectivity", False, f"Problème réseau: {str(e)}")

            except Exception as e:
                self.log_test("Exchange Setup", False, f"Configuration échouée: {str(e)}")

        except ImportError:
            self.log_test("CCXT Library", False, "Bibliothèque manquante", True)

    def test_performance_validation(self):
        """Test de validation des performances."""
        print("\n🚀 TESTS DE PERFORMANCE")
        print("=" * 60)

        # Test vitesse de chargement du modèle
        model_path = 'models/final_model.zip'
        if os.path.exists(model_path):
            start_time = time.time()
            try:
                model = PPO.load(model_path)
                load_time = time.time() - start_time

                if load_time < 10:
                    self.log_test("Model Load Speed", True, f"{load_time:.2f}s (< 10s)")
                else:
                    self.log_test("Model Load Speed", False, f"{load_time:.2f}s (> 10s)")

                # Test vitesse de prédiction
                test_obs = {
                    'image_features': np.random.randn(1, 1, 20, 235).astype(np.float32),
                    'vector_features': np.random.randn(1, 6).astype(np.float32)
                }

                prediction_times = []
                for _ in range(10):
                    start = time.time()
                    action, _ = model.predict(test_obs, deterministic=True)
                    prediction_times.append(time.time() - start)

                avg_pred_time = np.mean(prediction_times) * 1000  # en ms
                if avg_pred_time < 100:  # < 100ms
                    self.log_test("Prediction Speed", True, f"{avg_pred_time:.1f}ms/prediction")
                else:
                    self.log_test("Prediction Speed", False, f"{avg_pred_time:.1f}ms/prediction (> 100ms)")

            except Exception as e:
                self.log_test("Model Load Speed", False, f"Erreur: {str(e)}")

    def run_operational_tests(self):
        """Exécute tous les tests opérationnels."""
        print("🎯 TESTS OPÉRATIONNELS FINAUX - SYSTÈME ADAN")
        print("=" * 80)
        print(f"Démarré le: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Environnement: {os.getcwd()}")

        # Exécution séquentielle des tests
        test_functions = [
            self.test_system_requirements,
            self.test_configuration_integrity,
            self.test_data_availability,
            self.test_model_functionality,
            self.test_environment_functionality,
            self.test_trading_logic,
            self.test_script_execution,
            self.test_exchange_readiness,
            self.test_performance_validation
        ]

        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
                self.log_test(f"Test {test_name}", False, f"Exception: {str(e)}", True)

        self.generate_final_report()

    def generate_final_report(self):
        """Génère le rapport final."""
        print("\n" + "=" * 80)
        print("📊 RAPPORT OPÉRATIONNEL FINAL")
        print("=" * 80)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['success'])
        failed_tests = total_tests - passed_tests
        critical_failed = len(self.critical_errors)

        # Statistiques
        print(f"🔍 Tests Exécutés: {total_tests}")
        print(f"✅ Tests Réussis: {passed_tests}")
        print(f"❌ Tests Échoués: {failed_tests}")
        print(f"🚨 Erreurs Critiques: {critical_failed}")

        # Score de réussite
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"📈 Taux de Réussite: {success_rate:.1f}%")

        # Temps d'exécution
        duration = datetime.now() - self.start_time
        print(f"⏱️ Durée d'Exécution: {duration.total_seconds():.1f}s")

        # Verdict final
        print("\n" + "=" * 80)
        if critical_failed == 0 and success_rate >= 90:
            print("🎉 SYSTÈME OPÉRATIONNEL - PRÊT POUR PRODUCTION")
            print("✨ Tous les tests critiques sont passés")
            print("🚀 Le système peut être déployé en toute sécurité")

            print("\n💡 PROCHAINES ÉTAPES:")
            print("   1. Configurer les clés API pour trading testnet/live")
            print("   2. Effectuer des tests de trading avec petit capital")
            print("   3. Monitorer les performances en temps réel")
            print("   4. Mettre en place l'alerting automatique")

        elif critical_failed == 0:
            print("⚠️ SYSTÈME FONCTIONNEL - CORRECTIONS MINEURES REQUISES")
            print("✅ Aucune erreur critique détectée")
            print("🔧 Quelques ajustements recommandés avant production")

        else:
            print("🚨 SYSTÈME NON OPÉRATIONNEL - CORRECTIONS CRITIQUES REQUISES")
            print(f"❌ {critical_failed} erreur(s) critique(s) détectée(s)")
            print("🛠️ Corrections obligatoires avant mise en production")

            print("\n🚨 ERREURS CRITIQUES À CORRIGER:")
            for error in self.critical_errors:
                details = self.results[error].get('details', '')
                print(f"   - {error}: {details}")

        print("\n" + "=" * 80)
        print("📋 RÉSUMÉ DÉTAILLÉ:")
        for test_name, result in self.results.items():
            status = "✅" if result['success'] else ("🚨" if result['critical'] else "❌")
            print(f"   {status} {test_name}: {result['details']}")

        return critical_failed == 0 and success_rate >= 90

if __name__ == "__main__":
    tester = OperationalTester()
    success = tester.run_operational_tests()
    sys.exit(0 if success else 1)
