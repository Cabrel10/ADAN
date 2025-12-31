#!/usr/bin/env python3
"""
Walk-Forward Validation
Valide les modèles existants sur des fenêtres glissantes de données
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("WALK-FORWARD VALIDATION")
print("="*70)

class WalkForwardValidator:
    """Valide les modèles sur des fenêtres glissantes"""
    
    def __init__(self, train_months=3, test_months=1, num_windows=6):
        self.train_months = train_months
        self.test_months = test_months
        self.num_windows = num_windows
        self.results = {}
    
    def load_data(self):
        """Charger les données"""
        print("\n1️⃣  Chargement des données...")
        try:
            data_paths = {
                '5m': "data/processed/indicators/train/BTCUSDT/5m.parquet",
                '1h': "data/processed/indicators/train/BTCUSDT/1h.parquet",
                '4h': "data/processed/indicators/train/BTCUSDT/4h.parquet"
            }
            
            data = {}
            for tf, path_str in data_paths.items():
                path = Path(path_str)
                if path.exists():
                    df = pd.read_parquet(path)
                    data[tf] = df
                    print(f"   ✅ {tf}: {len(df)} barres chargées")
                else:
                    print(f"   ⚠️  {tf}: Données non trouvées")
                    return None
            
            return data
        except Exception as e:
            print(f"   ❌ Erreur chargement: {e}")
            return None
    
    def create_windows(self, data):
        """Créer des fenêtres glissantes"""
        print(f"\n2️⃣  Création de {self.num_windows} fenêtres glissantes...")
        try:
            # Utiliser le nombre de barres 5m comme référence
            total_bars = len(data['5m'])
            bars_per_month = total_bars // 6  # Approximation: 6 mois de données
            
            train_size = bars_per_month * self.train_months
            test_size = bars_per_month * self.test_months
            step_size = bars_per_month * self.test_months
            
            windows = []
            for i in range(self.num_windows):
                start_idx = i * step_size
                train_end = start_idx + train_size
                test_end = train_end + test_size
                
                if test_end > total_bars:
                    break
                
                window = {
                    'window_id': i + 1,
                    'train_start': start_idx,
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': test_end
                }
                windows.append(window)
                
                print(f"   ✅ Window {i+1}: Train [{start_idx}:{train_end}] Test [{train_end}:{test_end}]")
            
            return windows
        except Exception as e:
            print(f"   ❌ Erreur création fenêtres: {e}")
            return None
    
    def test_worker_on_window(self, worker_id, data, window):
        """Tester un worker sur une fenêtre"""
        try:
            from scripts.paper_trading_monitor import RealPaperTradingMonitor
            from stable_baselines3 import PPO
            
            # Charger le modèle
            model_paths = [
                Path(f"models/{worker_id}/model.zip"),
                Path(f"models/{worker_id}_final.zip"),
                Path(f"/mnt/new_data/t10_training/checkpoints/final/{worker_id}_final.zip")
            ]
            
            model_path = None
            for p in model_paths:
                if p.exists():
                    model_path = p
                    break
            
            if model_path is None:
                return None
            
            model = PPO.load(str(model_path))
            monitor = RealPaperTradingMonitor(api_key="test", api_secret="test")
            
            # Tester sur la fenêtre de test
            test_start = window['test_start']
            test_end = window['test_end']
            
            actions_list = []
            num_iterations = min(test_end - test_start, 50)
            
            for i in range(num_iterations):
                try:
                    idx = test_start + i
                    
                    raw_data = {
                        'BTC/USDT': {
                            '5m': data['5m'].iloc[max(0, idx-20):idx+1],
                            '1h': data['1h'].iloc[max(0, idx-20):idx+1],
                            '4h': data['4h'].iloc[max(0, idx-20):idx+1]
                        }
                    }
                    
                    obs = monitor.build_observation(worker_id, raw_data)
                    if obs is None:
                        continue
                    
                    action, _ = model.predict(obs, deterministic=True)
                    actions_list.append(float(action.mean()))
                except:
                    continue
            
            if len(actions_list) > 0:
                actions = np.array(actions_list)
                return {
                    'num_actions': len(actions_list),
                    'mean': float(actions.mean()),
                    'std': float(actions.std()),
                    'min': float(actions.min()),
                    'max': float(actions.max())
                }
            else:
                return None
        except Exception as e:
            return None
    
    def validate_windows(self, data, windows):
        """Valider les modèles sur toutes les fenêtres"""
        print(f"\n3️⃣  Validation sur {len(windows)} fenêtres...")
        
        for worker_id in ["w1", "w2", "w3", "w4"]:
            print(f"\n   {worker_id}:")
            
            window_results = []
            for window in windows:
                result = self.test_worker_on_window(worker_id, data, window)
                if result:
                    window_results.append(result)
                    print(f"      ✅ Window {window['window_id']}: Mean={result['mean']:.4f}, Std={result['std']:.4f}")
                else:
                    print(f"      ⚠️  Window {window['window_id']}: Pas de résultats")
            
            if len(window_results) > 0:
                # Calculer la stabilité
                means = np.array([r['mean'] for r in window_results])
                stability = float(np.std(means))
                
                self.results[worker_id] = {
                    'num_windows': len(window_results),
                    'window_results': window_results,
                    'stability': stability,
                    'status': 'STABLE' if stability < 0.1 else 'UNSTABLE'
                }
                
                print(f"      Stabilité: {stability:.4f} ({self.results[worker_id]['status']})")
    
    def generate_report(self):
        """Générer le rapport"""
        print("\n4️⃣  Génération du rapport...")
        try:
            # Compter les statuts
            stable = sum(1 for r in self.results.values() if r['status'] == 'STABLE')
            unstable = sum(1 for r in self.results.values() if r['status'] == 'UNSTABLE')
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'test': 'walk_forward_validation',
                'parameters': {
                    'train_months': self.train_months,
                    'test_months': self.test_months,
                    'num_windows': self.num_windows
                },
                'results': self.results,
                'summary': {
                    'total_workers': len(self.results),
                    'stable_workers': stable,
                    'unstable_workers': unstable,
                    'overall_status': 'PASSED' if stable >= 3 else 'FAILED'
                }
            }
            
            # Sauvegarder
            output_path = Path("diagnostic/results/walk_forward_validation.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"   ✅ Rapport sauvegardé: {output_path}")
            return report
        except Exception as e:
            print(f"   ❌ Erreur génération rapport: {e}")
            return None
    
    def run_validation(self):
        """Exécuter la validation complète"""
        # Charger les données
        data = self.load_data()
        if data is None:
            return False
        
        # Créer les fenêtres
        windows = self.create_windows(data)
        if windows is None or len(windows) == 0:
            return False
        
        # Valider sur les fenêtres
        self.validate_windows(data, windows)
        
        # Générer le rapport
        report = self.generate_report()
        
        # Affichage final
        print("\n" + "="*70)
        if report and report['summary']['overall_status'] == 'PASSED':
            print("✅ WALK-FORWARD VALIDATION RÉUSSIE")
            print("="*70)
            print(f"\nRésumé:")
            print(f"  - Workers stables: {report['summary']['stable_workers']}/4")
            print(f"  - Workers instables: {report['summary']['unstable_workers']}/4")
            print(f"  - Status: {report['summary']['overall_status']}")
            print("\nProchaine étape: Paper Trading Extended")
            return True
        else:
            print("❌ WALK-FORWARD VALIDATION ÉCHOUÉE")
            print("="*70)
            if report:
                print(f"\nRésumé:")
                print(f"  - Workers stables: {report['summary']['stable_workers']}/4")
                print(f"  - Workers instables: {report['summary']['unstable_workers']}/4")
            return False

def main():
    """Fonction principale"""
    validator = WalkForwardValidator(train_months=3, test_months=1, num_windows=6)
    success = validator.run_validation()
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
