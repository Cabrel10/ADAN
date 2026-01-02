#!/usr/bin/env python3
"""
Checkpoint 3.2: Paper Trading Dry-Run
Exécute 100 itérations de paper trading sans appels API réels.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("CHECKPOINT 3.2: Paper Trading Dry-Run")
print("="*70)

class PaperTradingDryRun:
    """Classe pour exécuter un dry-run de paper trading"""
    
    def __init__(self, num_iterations: int = 100):
        self.num_iterations = num_iterations
        self.monitor = None
        self.portfolio = None
        self.decisions = []
        self.statistics = {
            'total_iterations': 0,
            'successful_iterations': 0,
            'failed_iterations': 0,
            'total_time': 0,
            'avg_time_per_iteration': 0,
            'errors': []
        }
    
    def initialize_monitor(self):
        """Initialiser le monitor"""
        print("\n1️⃣  Initialisation du monitor...")
        try:
            from scripts.paper_trading_monitor import RealPaperTradingMonitor
            
            self.monitor = RealPaperTradingMonitor(
                api_key="test_key",
                api_secret="test_secret"
            )
            
            # Initialiser les environnements de normalisation
            self.monitor.initialize_worker_environments()
            
            print("   ✅ Monitor initialisé")
            return True
        except Exception as e:
            print(f"   ❌ Erreur initialisation: {e}")
            self.statistics['errors'].append(f"Monitor init: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_portfolio(self):
        """Initialiser l'état du portfolio"""
        print("\n2️⃣  Initialisation du portfolio...")
        try:
            self.portfolio = {
                'cash': 10000.0,
                'positions': {},
                'total_value': 10000.0,
                'iteration': 0,
                'timestamp': datetime.now().isoformat()
            }
            print(f"   ✅ Portfolio initialisé (cash: ${self.portfolio['cash']:.2f})")
            return True
        except Exception as e:
            print(f"   ❌ Erreur initialisation portfolio: {e}")
            self.statistics['errors'].append(f"Portfolio init: {str(e)}")
            return False
    
    def load_market_data(self):
        """Charger les données de marché"""
        try:
            # Créer des données simulées avec les bonnes colonnes et shapes
            # Les colonnes doivent avoir des noms (pas des indices numériques)
            # 14 features pour correspondre aux shapes attendues
            columns = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'macd_12_26_9', 
                      'bb_percent_b_20_2', 'atr_14', 'atr_20', 'atr_50', 'volume_ratio_20', 
                      'ema_20_ratio', 'stoch_k_14_3_3']
            
            raw_data = {
                'BTC/USDT': {
                    '5m': pd.DataFrame(np.random.randn(20, 14), columns=columns),
                    '1h': pd.DataFrame(np.random.randn(20, 14), columns=columns),
                    '4h': pd.DataFrame(np.random.randn(20, 14), columns=columns)
                }
            }
            
            return raw_data
        except Exception as e:
            print(f"   ❌ Erreur chargement données: {e}")
            return None
    
    def run_iteration(self, iteration: int):
        """Exécuter une itération de trading"""
        iteration_start = time.time()
        
        try:
            # Charger les données
            raw_data = self.load_market_data()
            if raw_data is None:
                raise Exception("Données de marché non disponibles")
            
            # Générer observations et prédictions pour chaque worker
            iteration_decisions = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'workers': {}
            }
            
            for worker_id in ["w1", "w2", "w3", "w4"]:
                try:
                    # Générer observation
                    obs = self.monitor.build_observation(worker_id, raw_data)
                    
                    if obs is None:
                        raise Exception(f"build_observation retourned None")
                    
                    # Obtenir prédiction
                    model = self.monitor.models[worker_id]
                    action, _states = model.predict(obs, deterministic=True)
                    
                    # Valider l'action
                    if np.isnan(action).any():
                        raise Exception(f"Action contient NaN")
                    
                    if action.min() < -1.1 or action.max() > 1.1:
                        raise Exception(f"Action hors limites: [{action.min():.3f}, {action.max():.3f}]")
                    
                    # Enregistrer la décision
                    iteration_decisions['workers'][worker_id] = {
                        'action': action.tolist() if hasattr(action, 'tolist') else action,
                        'action_mean': float(action.mean()),
                        'action_std': float(action.std()),
                        'action_min': float(action.min()),
                        'action_max': float(action.max()),
                        'valid': True
                    }
                    
                except Exception as e:
                    iteration_decisions['workers'][worker_id] = {
                        'valid': False,
                        'error': str(e)
                    }
            
            # Mettre à jour le portfolio (simulation simple)
            self.portfolio['iteration'] = iteration
            self.portfolio['timestamp'] = datetime.now().isoformat()
            
            # Enregistrer la décision
            self.decisions.append(iteration_decisions)
            
            # Calculer le temps d'exécution
            iteration_time = time.time() - iteration_start
            
            # Vérifier le succès
            successful_workers = sum(1 for w in iteration_decisions['workers'].values() if w.get('valid', False))
            
            if successful_workers >= 3:  # Au moins 3/4 workers
                self.statistics['successful_iterations'] += 1
                status = "✅"
            else:
                self.statistics['failed_iterations'] += 1
                status = "⚠️"
            
            self.statistics['total_time'] += iteration_time
            
            # Afficher le progrès tous les 10 itérations
            if (iteration + 1) % 10 == 0:
                print(f"   {status} Itération {iteration + 1}/100 ({successful_workers}/4 workers) - {iteration_time:.3f}s")
            
            return True
            
        except Exception as e:
            self.statistics['failed_iterations'] += 1
            self.statistics['errors'].append(f"Iteration {iteration}: {str(e)}")
            print(f"   ❌ Erreur itération {iteration}: {e}")
            return False
    
    def run_dry_run(self):
        """Exécuter le dry-run complet"""
        print("\n3️⃣  Exécution du dry-run (100 itérations)...")
        
        start_time = time.time()
        
        for iteration in range(self.num_iterations):
            self.run_iteration(iteration)
        
        total_time = time.time() - start_time
        self.statistics['total_iterations'] = self.num_iterations
        self.statistics['total_time'] = total_time
        self.statistics['avg_time_per_iteration'] = total_time / self.num_iterations
        
        print(f"\n   ✅ Dry-run complété en {total_time:.2f}s")
        return True
    
    def collect_statistics(self):
        """Collecter les statistiques"""
        print("\n4️⃣  Collecte des statistiques...")
        
        try:
            # Statistiques globales
            total_decisions = len(self.decisions)
            successful_decisions = sum(1 for d in self.decisions 
                                      if sum(1 for w in d['workers'].values() if w.get('valid', False)) >= 3)
            
            # Statistiques par worker
            worker_stats = {}
            for worker_id in ["w1", "w2", "w3", "w4"]:
                valid_count = 0
                actions_list = []
                
                for decision in self.decisions:
                    if worker_id in decision['workers']:
                        worker_data = decision['workers'][worker_id]
                        if worker_data.get('valid', False):
                            valid_count += 1
                            actions_list.append(worker_data['action'])
                
                if actions_list:
                    actions_array = np.array(actions_list)
                    worker_stats[worker_id] = {
                        'valid_predictions': valid_count,
                        'total_predictions': len(self.decisions),
                        'success_rate': valid_count / len(self.decisions),
                        'action_mean': float(actions_array.mean()),
                        'action_std': float(actions_array.std()),
                        'action_min': float(actions_array.min()),
                        'action_max': float(actions_array.max())
                    }
            
            self.statistics['worker_stats'] = worker_stats
            self.statistics['total_decisions'] = total_decisions
            self.statistics['successful_decisions'] = successful_decisions
            
            print(f"   ✅ Statistiques collectées")
            print(f"      - Total décisions: {total_decisions}")
            print(f"      - Décisions réussies: {successful_decisions}")
            print(f"      - Taux de succès: {successful_decisions/total_decisions*100:.1f}%")
            
            return True
        except Exception as e:
            print(f"   ❌ Erreur collecte statistiques: {e}")
            self.statistics['errors'].append(f"Statistics: {str(e)}")
            return False
    
    def generate_report(self):
        """Générer le rapport"""
        print("\n5️⃣  Génération du rapport...")
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'checkpoint': '3.2',
                'test': 'paper_trading_dryrun',
                'portfolio': self.portfolio,
                'statistics': self.statistics,
                'decisions_sample': self.decisions[:5] if self.decisions else [],  # Premiers 5 pour l'exemple
                'summary': {
                    'total_iterations': self.statistics['total_iterations'],
                    'successful_iterations': self.statistics['successful_iterations'],
                    'failed_iterations': self.statistics['failed_iterations'],
                    'success_rate': self.statistics['successful_iterations'] / self.statistics['total_iterations'] if self.statistics['total_iterations'] > 0 else 0,
                    'avg_time_per_iteration': self.statistics['avg_time_per_iteration'],
                    'total_time': self.statistics['total_time']
                }
            }
            
            # Sauvegarder le rapport
            output_path = Path("diagnostic/results/checkpoint_3_2_results.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"   ✅ Rapport sauvegardé: {output_path}")
            return report
        except Exception as e:
            print(f"   ❌ Erreur génération rapport: {e}")
            self.statistics['errors'].append(f"Report: {str(e)}")
            return None

def main():
    """Fonction principale"""
    
    # Créer l'instance de dry-run
    dry_run = PaperTradingDryRun(num_iterations=100)
    
    # Initialiser le monitor
    if not dry_run.initialize_monitor():
        print("\n❌ CHECKPOINT 3.2 ÉCHOUÉ - Monitor non initialisé")
        return False
    
    # Initialiser le portfolio
    if not dry_run.initialize_portfolio():
        print("\n❌ CHECKPOINT 3.2 ÉCHOUÉ - Portfolio non initialisé")
        return False
    
    # Exécuter le dry-run
    if not dry_run.run_dry_run():
        print("\n❌ CHECKPOINT 3.2 ÉCHOUÉ - Dry-run échoué")
        return False
    
    # Collecter les statistiques
    if not dry_run.collect_statistics():
        print("\n❌ CHECKPOINT 3.2 ÉCHOUÉ - Statistiques non collectées")
        return False
    
    # Générer le rapport
    report = dry_run.generate_report()
    if report is None:
        print("\n❌ CHECKPOINT 3.2 ÉCHOUÉ - Rapport non généré")
        return False
    
    # Évaluation finale
    print("\n" + "="*70)
    success_rate = report['summary']['success_rate']
    
    if success_rate >= 0.95:  # Au moins 95% de succès
        print("✅ CHECKPOINT 3.2 VALIDÉ")
        print("="*70)
        print(f"\nRésultats:")
        print(f"  - Itérations complétées: {report['summary']['total_iterations']}")
        print(f"  - Itérations réussies: {report['summary']['successful_iterations']}")
        print(f"  - Taux de succès: {success_rate*100:.1f}%")
        print(f"  - Temps moyen par itération: {report['summary']['avg_time_per_iteration']:.3f}s")
        print(f"  - Temps total: {report['summary']['total_time']:.2f}s")
        print("\nProchaine étape: Checkpoint 3.3 - Analyse des Décisions")
        return True
    else:
        print("❌ CHECKPOINT 3.2 ÉCHOUÉ")
        print("="*70)
        print(f"\nRésultats:")
        print(f"  - Taux de succès: {success_rate*100:.1f}% (minimum: 95%)")
        print(f"  - Erreurs: {len(dry_run.statistics['errors'])}")
        for error in dry_run.statistics['errors'][:5]:
            print(f"    - {error}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
