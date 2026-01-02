#!/usr/bin/env python3
"""
Checkpoint 3.2: Paper Trading Dry-Run (Version 2)
Exécute 100 itérations de paper trading en utilisant les données réelles du test 3.1.
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
print("CHECKPOINT 3.2: Paper Trading Dry-Run (Version 2)")
print("="*70)

def main():
    """Fonction principale"""
    
    print("\n1️⃣  Initialisation du monitor...")
    try:
        from scripts.paper_trading_monitor import RealPaperTradingMonitor
        
        monitor = RealPaperTradingMonitor(
            api_key="test_key",
            api_secret="test_secret"
        )
        
        # Initialiser le pipeline (charge les modèles)
        if not monitor.setup_pipeline():
            raise Exception("Pipeline setup failed")
        
        print("   ✅ Monitor initialisé")
    except Exception as e:
        print(f"   ❌ Erreur initialisation: {e}")
        return False
    
    print("\n2️⃣  Initialisation du portfolio...")
    portfolio = {
        'cash': 10000.0,
        'positions': {},
        'total_value': 10000.0,
        'iteration': 0,
        'timestamp': datetime.now().isoformat()
    }
    print(f"   ✅ Portfolio initialisé (cash: ${portfolio['cash']:.2f})")
    
    print("\n3️⃣  Exécution du dry-run (100 itérations)...")
    
    decisions = []
    statistics = {
        'total_iterations': 100,
        'successful_iterations': 0,
        'failed_iterations': 0,
        'total_time': 0,
        'avg_time_per_iteration': 0,
        'errors': []
    }
    
    start_time = time.time()
    
    for iteration in range(100):
        iteration_start = time.time()
        
        try:
            # Charger les données réelles
            data_paths = {
                '5m': "data/processed/indicators/train/BTCUSDT/5m.parquet",
                '1h': "data/processed/indicators/train/BTCUSDT/1h.parquet",
                '4h': "data/processed/indicators/train/BTCUSDT/4h.parquet"
            }
            
            raw_data = {}
            for tf, path_str in data_paths.items():
                path = Path(path_str)
                if path.exists():
                    df = pd.read_parquet(path)
                    if tf == '5m':
                        raw_data[tf] = df.tail(20)
                    elif tf == '1h':
                        raw_data[tf] = df.tail(20)  # Utiliser 20 au lieu de 10
                    elif tf == '4h':
                        raw_data[tf] = df.tail(20)  # Utiliser 20 au lieu de 5
                else:
                    raise Exception(f"Données manquantes: {path}")
            
            # Générer observations et prédictions pour chaque worker
            iteration_decisions = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'workers': {}
            }
            
            for worker_id in ["w1", "w2", "w3", "w4"]:
                try:
                    # Générer observation
                    obs = monitor.build_observation(worker_id, {'BTC/USDT': raw_data})
                    
                    if obs is None:
                        raise Exception(f"build_observation retourned None")
                    
                    # Obtenir prédiction
                    model = monitor.workers[worker_id]
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
            
            # Mettre à jour le portfolio
            portfolio['iteration'] = iteration
            portfolio['timestamp'] = datetime.now().isoformat()
            
            # Enregistrer la décision
            decisions.append(iteration_decisions)
            
            # Calculer le temps d'exécution
            iteration_time = time.time() - iteration_start
            
            # Vérifier le succès
            successful_workers = sum(1 for w in iteration_decisions['workers'].values() if w.get('valid', False))
            
            if successful_workers >= 3:  # Au moins 3/4 workers
                statistics['successful_iterations'] += 1
                status = "✅"
            else:
                statistics['failed_iterations'] += 1
                status = "⚠️"
            
            statistics['total_time'] += iteration_time
            
            # Afficher le progrès tous les 10 itérations
            if (iteration + 1) % 10 == 0:
                print(f"   {status} Itération {iteration + 1}/100 ({successful_workers}/4 workers) - {iteration_time:.3f}s")
            
        except Exception as e:
            statistics['failed_iterations'] += 1
            statistics['errors'].append(f"Iteration {iteration}: {str(e)}")
            print(f"   ❌ Erreur itération {iteration}: {e}")
    
    total_time = time.time() - start_time
    statistics['total_time'] = total_time
    statistics['avg_time_per_iteration'] = total_time / 100
    
    print(f"\n   ✅ Dry-run complété en {total_time:.2f}s")
    
    print("\n4️⃣  Collecte des statistiques...")
    
    # Statistiques globales
    total_decisions = len(decisions)
    successful_decisions = sum(1 for d in decisions 
                              if sum(1 for w in d['workers'].values() if w.get('valid', False)) >= 3)
    
    # Statistiques par worker
    worker_stats = {}
    for worker_id in ["w1", "w2", "w3", "w4"]:
        valid_count = 0
        actions_list = []
        
        for decision in decisions:
            if worker_id in decision['workers']:
                worker_data = decision['workers'][worker_id]
                if worker_data.get('valid', False):
                    valid_count += 1
                    actions_list.append(worker_data['action'])
        
        if actions_list:
            actions_array = np.array(actions_list)
            worker_stats[worker_id] = {
                'valid_predictions': valid_count,
                'total_predictions': len(decisions),
                'success_rate': valid_count / len(decisions),
                'action_mean': float(actions_array.mean()),
                'action_std': float(actions_array.std()),
                'action_min': float(actions_array.min()),
                'action_max': float(actions_array.max())
            }
    
    statistics['worker_stats'] = worker_stats
    statistics['total_decisions'] = total_decisions
    statistics['successful_decisions'] = successful_decisions
    
    print(f"   ✅ Statistiques collectées")
    print(f"      - Total décisions: {total_decisions}")
    print(f"      - Décisions réussies: {successful_decisions}")
    print(f"      - Taux de succès: {successful_decisions/total_decisions*100:.1f}%")
    
    print("\n5️⃣  Génération du rapport...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': '3.2',
        'test': 'paper_trading_dryrun',
        'portfolio': portfolio,
        'statistics': statistics,
        'decisions_sample': decisions[:5] if decisions else [],
        'summary': {
            'total_iterations': statistics['total_iterations'],
            'successful_iterations': statistics['successful_iterations'],
            'failed_iterations': statistics['failed_iterations'],
            'success_rate': statistics['successful_iterations'] / statistics['total_iterations'] if statistics['total_iterations'] > 0 else 0,
            'avg_time_per_iteration': statistics['avg_time_per_iteration'],
            'total_time': statistics['total_time']
        }
    }
    
    # Sauvegarder le rapport
    output_path = Path("diagnostic/results/checkpoint_3_2_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✅ Rapport sauvegardé: {output_path}")
    
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
        print(f"  - Erreurs: {len(statistics['errors'])}")
        for error in statistics['errors'][:5]:
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
