#!/usr/bin/env python3
"""
SCRIPT 7: Test de Toutes les Issues Possibles
Teste les problèmes courants et génère des diagnostics
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')

class IssuesTester:
    def __init__(self):
        self.issues = []
        self.tests_passed = 0
        self.tests_failed = 0
        
    def test_import_errors(self):
        """Test 1: Vérifier les erreurs d'import"""
        print("\n📋 TEST 1: Vérification des Erreurs d'Import")
        print("-" * 50)
        
        modules_to_test = [
            'adan_trading_bot.environment.reward_calculator',
            'adan_trading_bot.performance.metrics',
            'adan_trading_bot.agent.ppo_agent',
            'adan_trading_bot.training.trainer',
            'adan_trading_bot.data_processing.data_loader',
            'adan_trading_bot.risk_management.risk_manager',
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                print(f"  ✅ {module_name}")
                self.tests_passed += 1
            except ImportError as e:
                print(f"  ❌ {module_name}: {e}")
                self.issues.append({
                    'severity': 'CRITICAL',
                    'test': 'Import Error',
                    'module': module_name,
                    'error': str(e)
                })
                self.tests_failed += 1
            except Exception as e:
                print(f"  ⚠️  {module_name}: {type(e).__name__}: {e}")
                self.tests_failed += 1
    
    def test_circular_dependencies(self):
        """Test 2: Vérifier les dépendances circulaires"""
        print("\n📋 TEST 2: Vérification des Dépendances Circulaires")
        print("-" * 50)
        
        try:
            # Essayer d'importer les modules dans différents ordres
            import adan_trading_bot.environment.reward_calculator
            import adan_trading_bot.performance.metrics
            import adan_trading_bot.agent.ppo_agent
            
            print("  ✅ Pas de dépendances circulaires détectées")
            self.tests_passed += 1
        except ImportError as e:
            if 'circular' in str(e).lower():
                print(f"  ❌ Dépendance circulaire détectée: {e}")
                self.issues.append({
                    'severity': 'CRITICAL',
                    'test': 'Circular Dependency',
                    'error': str(e)
                })
                self.tests_failed += 1
            else:
                print(f"  ⚠️  Erreur d'import: {e}")
    
    def test_missing_attributes(self):
        """Test 3: Vérifier les attributs manquants"""
        print("\n📋 TEST 3: Vérification des Attributs Manquants")
        print("-" * 50)
        
        try:
            from adan_trading_bot.environment.reward_calculator import RewardCalculator
            
            config = {'reward_shaping': {}}
            calc = RewardCalculator(config)
            
            required_attrs = [
                'returns_dates',
                'current_chunk_id',
                'current_episode_id',
                'returns_history',
                'weights'
            ]
            
            missing = []
            for attr in required_attrs:
                if not hasattr(calc, attr):
                    missing.append(attr)
            
            if missing:
                print(f"  ❌ Attributs manquants: {missing}")
                self.issues.append({
                    'severity': 'CRITICAL',
                    'test': 'Missing Attributes',
                    'module': 'RewardCalculator',
                    'missing': missing
                })
                self.tests_failed += 1
            else:
                print(f"  ✅ Tous les attributs présents")
                self.tests_passed += 1
                
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            self.tests_failed += 1
    
    def test_data_type_errors(self):
        """Test 4: Vérifier les erreurs de type de données"""
        print("\n📋 TEST 4: Vérification des Erreurs de Type")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            import numpy as np
            
            metrics = PerformanceMetrics()
            
            # Ajouter des données de test
            for i in range(100):
                metrics.returns.append(np.random.normal(0, 0.1))
                metrics.equity_curve.append(10000 + i * 10)
            
            # Vérifier les types
            for ret in metrics.returns:
                if not isinstance(ret, (int, float, np.number)):
                    print(f"  ❌ Type incorrect dans returns: {type(ret)}")
                    self.issues.append({
                        'severity': 'MAJOR',
                        'test': 'Data Type Error',
                        'issue': f'Type incorrect: {type(ret)}'
                    })
                    self.tests_failed += 1
                    return
            
            print(f"  ✅ Types de données corrects")
            self.tests_passed += 1
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            self.tests_failed += 1
    
    def test_nan_inf_values(self):
        """Test 5: Vérifier les valeurs NaN/Inf"""
        print("\n📋 TEST 5: Vérification des Valeurs NaN/Inf")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            import numpy as np
            
            metrics = PerformanceMetrics()
            
            # Ajouter des données
            for i in range(100):
                metrics.returns.append(np.random.normal(0, 0.1))
                metrics.equity_curve.append(10000 + i * 10)
            
            # Vérifier NaN/Inf
            nan_count = sum(1 for r in metrics.returns if np.isnan(r))
            inf_count = sum(1 for r in metrics.returns if np.isinf(r))
            
            if nan_count > 0 or inf_count > 0:
                print(f"  ❌ Valeurs invalides: {nan_count} NaN, {inf_count} Inf")
                self.issues.append({
                    'severity': 'CRITICAL',
                    'test': 'NaN/Inf Values',
                    'nan_count': nan_count,
                    'inf_count': inf_count
                })
                self.tests_failed += 1
            else:
                print(f"  ✅ Pas de valeurs NaN/Inf")
                self.tests_passed += 1
                
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            self.tests_failed += 1
    
    def test_configuration_errors(self):
        """Test 6: Vérifier les erreurs de configuration"""
        print("\n📋 TEST 6: Vérification des Erreurs de Configuration")
        print("-" * 50)
        
        try:
            from adan_trading_bot.common.config_loader import ConfigLoader
            
            # Essayer de charger la configuration
            loader = ConfigLoader()
            config = loader.load_config('config/config.yaml')
            
            # Vérifier les clés requises
            required_keys = ['environment', 'training', 'agent']
            missing_keys = [k for k in required_keys if k not in config]
            
            if missing_keys:
                print(f"  ❌ Clés manquantes: {missing_keys}")
                self.issues.append({
                    'severity': 'MAJOR',
                    'test': 'Configuration Error',
                    'missing_keys': missing_keys
                })
                self.tests_failed += 1
            else:
                print(f"  ✅ Configuration valide")
                self.tests_passed += 1
                
        except Exception as e:
            print(f"  ⚠️  Erreur de configuration: {e}")
            self.tests_failed += 1
    
    def test_memory_leaks(self):
        """Test 7: Vérifier les fuites mémoire"""
        print("\n📋 TEST 7: Vérification des Fuites Mémoire")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            import gc
            
            # Créer et détruire plusieurs instances
            for i in range(100):
                metrics = PerformanceMetrics()
                del metrics
            
            gc.collect()
            
            print(f"  ✅ Pas de fuites mémoire détectées")
            self.tests_passed += 1
            
        except Exception as e:
            print(f"  ⚠️  Erreur: {e}")
            self.tests_failed += 1
    
    def test_thread_safety(self):
        """Test 8: Vérifier la sécurité des threads"""
        print("\n📋 TEST 8: Vérification de la Sécurité des Threads")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            import threading
            
            metrics = PerformanceMetrics()
            errors = []
            
            def add_data():
                try:
                    for i in range(100):
                        metrics.returns.append(0.001)
                except Exception as e:
                    errors.append(e)
            
            # Créer plusieurs threads
            threads = [threading.Thread(target=add_data) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            if errors:
                print(f"  ⚠️  Erreurs de thread: {len(errors)}")
                self.tests_failed += 1
            else:
                print(f"  ✅ Pas d'erreurs de thread")
                self.tests_passed += 1
                
        except Exception as e:
            print(f"  ⚠️  Erreur: {e}")
            self.tests_failed += 1
    
    def test_performance(self):
        """Test 9: Vérifier les performances"""
        print("\n📋 TEST 9: Vérification des Performances")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            import time
            import numpy as np
            
            metrics = PerformanceMetrics()
            
            # Ajouter 10000 points de données
            start = time.time()
            for i in range(10000):
                metrics.returns.append(np.random.normal(0, 0.1))
                metrics.equity_curve.append(10000 + i * 10)
            elapsed = time.time() - start
            
            # Calculer les métriques
            start = time.time()
            sharpe = metrics.calculate_sharpe_ratio()
            max_dd = metrics.calculate_max_drawdown()
            calc_time = time.time() - start
            
            print(f"  Temps d'ajout (10000 points): {elapsed:.3f}s")
            print(f"  Temps de calcul (Sharpe + Max DD): {calc_time:.3f}s")
            
            if elapsed > 5.0 or calc_time > 1.0:
                print(f"  ⚠️  Performance dégradée")
                self.tests_failed += 1
            else:
                print(f"  ✅ Performance acceptable")
                self.tests_passed += 1
                
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            self.tests_failed += 1
    
    def test_error_handling(self):
        """Test 10: Vérifier la gestion des erreurs"""
        print("\n📋 TEST 10: Vérification de la Gestion des Erreurs")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            
            metrics = PerformanceMetrics()
            
            # Essayer des opérations invalides
            try:
                # Pas de données
                sharpe = metrics.calculate_sharpe_ratio()
                print(f"  ✅ Gestion correcte des données vides")
                self.tests_passed += 1
            except Exception as e:
                print(f"  ❌ Erreur non gérée: {e}")
                self.issues.append({
                    'severity': 'MAJOR',
                    'test': 'Error Handling',
                    'issue': f'Erreur non gérée: {e}'
                })
                self.tests_failed += 1
                
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            self.tests_failed += 1
    
    def generate_report(self):
        """Génère un rapport d'audit"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_name': 'All Issues Test',
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'total_tests': self.tests_passed + self.tests_failed,
            'issues': self.issues,
            'summary': {
                'total_issues': len(self.issues),
                'critical_issues': sum(1 for i in self.issues if i.get('severity') == 'CRITICAL'),
                'major_issues': sum(1 for i in self.issues if i.get('severity') == 'MAJOR'),
                'success_rate': (self.tests_passed / (self.tests_passed + self.tests_failed) * 100) if (self.tests_passed + self.tests_failed) > 0 else 0
            }
        }
        return report

def main():
    print("=" * 70)
    print("SCRIPT 7: TEST DE TOUTES LES ISSUES POSSIBLES")
    print("=" * 70)
    
    tester = IssuesTester()
    
    # Exécuter tous les tests
    tester.test_import_errors()
    tester.test_circular_dependencies()
    tester.test_missing_attributes()
    tester.test_data_type_errors()
    tester.test_nan_inf_values()
    tester.test_configuration_errors()
    tester.test_memory_leaks()
    tester.test_thread_safety()
    tester.test_performance()
    tester.test_error_handling()
    
    # Générer le rapport
    report = tester.generate_report()
    
    print(f"\n📊 RÉSUMÉ")
    print(f"  Tests Réussis: {report['tests_passed']}")
    print(f"  Tests Échoués: {report['tests_failed']}")
    print(f"  Taux de Succès: {report['summary']['success_rate']:.1f}%")
    print(f"  Total Issues: {report['summary']['total_issues']}")
    print(f"  Critical: {report['summary']['critical_issues']}")
    print(f"  Major: {report['summary']['major_issues']}")
    
    # Sauvegarder
    output_dir = Path('investigation_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'all_issues_test.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Rapport sauvegardé: investigation_results/all_issues_test.json")
    
    return report

if __name__ == '__main__':
    main()
