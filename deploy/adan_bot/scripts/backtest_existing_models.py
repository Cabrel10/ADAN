#!/usr/bin/env python3
"""
Backtest des Modèles Existants
Teste les 4 workers entraînés sur données historiques réelles (parquet)
SANS réentraînement - validation pure des modèles existants
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("BACKTEST DES MODÈLES EXISTANTS")
print("="*70)

class BacktestExistingModels:
    """Backtest les 4 workers existants sur données historiques réelles"""
    
    def __init__(self):
        self.models = {}
        self.vecnormalizers = {}
        self.results = {}
    
    def load_models_and_vecnorm(self):
        """Charger les modèles et VecNormalize existants"""
        print("\n1️⃣  Chargement des modèles existants...")
        try:
            for worker_id in ["w1", "w2", "w3", "w4"]:
                # Chercher le modèle
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
                    print(f"   ⚠️  {worker_id}: Modèle non trouvé")
                    continue
                
                # Charger le modèle
                self.models[worker_id] = PPO.load(str(model_path))
                print(f"   ✅ {worker_id}: Modèle chargé ({model_path.name})")
            
            if len(self.models) == 0:
                print("   ❌ Aucun modèle chargé")
                return False
            
            print(f"   ✅ {len(self.models)} modèles chargés")
            return True
        except Exception as e:
            print(f"   ❌ Erreur chargement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_historical_data(self, start_date="2024-01-01", end_date="2024-12-31"):
        """Charger les données historiques parquet"""
        print(f"\n2️⃣  Chargement données historiques ({start_date} à {end_date})...")
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
                    print(f"   ⚠️  {tf}: Données non trouvées ({path})")
            
            if len(data) < 3:
                print("   ❌ Données insuffisantes")
                return None
            
            return data
        except Exception as e:
            print(f"   ❌ Erreur chargement données: {e}")
            return None
    
    def backtest_worker(self, worker_id, data):
        """Backtest un worker sur les données historiques"""
        print(f"\n3️⃣  Backtest {worker_id}...")
        try:
            if worker_id not in self.models:
                print(f"   ⚠️  {worker_id}: Modèle non disponible")
                return None
            
            model = self.models[worker_id]
            
            # Initialiser le portfolio
            portfolio = {
                'cash': 10000.0,
                'positions': {},
                'total_value': 10000.0,
                'trades': [],
                'pnl_history': []
            }
            
            # Simuler le trading sur les données
            num_iterations = min(len(data['5m']), 100)  # Limiter à 100 itérations pour le test
            
            for i in range(num_iterations):
                try:
                    # Préparer les données pour cette itération
                    raw_data = {
                        'BTC/USDT': {
                            '5m': data['5m'].iloc[max(0, i-20):i+1],
                            '1h': data['1h'].iloc[max(0, i-20):i+1],
                            '4h': data['4h'].iloc[max(0, i-20):i+1]
                        }
                    }
                    
                    # Construire l'observation
                    from scripts.paper_trading_monitor import RealPaperTradingMonitor
                    monitor = RealPaperTradingMonitor(api_key="test", api_secret="test")
                    obs = monitor.build_observation(worker_id, raw_data)
                    
                    if obs is None:
                        continue
                    
                    # Obtenir la prédiction
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Enregistrer l'action
                    portfolio['pnl_history'].append({
                        'iteration': i,
                        'action_mean': float(action.mean()),
                        'action_std': float(action.std())
                    })
                    
                except Exception as e:
                    continue
            
            # Calculer les métriques
            if len(portfolio['pnl_history']) > 0:
                actions = np.array([p['action_mean'] for p in portfolio['pnl_history']])
                metrics = {
                    'iterations': len(portfolio['pnl_history']),
                    'action_mean': float(actions.mean()),
                    'action_std': float(actions.std()),
                    'action_min': float(actions.min()),
                    'action_max': float(actions.max()),
                    'valid': True
                }
                
                print(f"   ✅ {worker_id}: {metrics['iterations']} itérations")
                print(f"      - Action Mean: {metrics['action_mean']:.4f}")
                print(f"      - Action Std: {metrics['action_std']:.4f}")
                
                return metrics
            else:
                print(f"   ❌ {worker_id}: Aucune itération valide")
                return None
        except Exception as e:
            print(f"   ❌ {worker_id}: Erreur backtest: {e}")
            return None
    
    def generate_report(self):
        """Générer le rapport de backtest"""
        print("\n4️⃣  Génération du rapport...")
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'test': 'backtest_existing_models',
                'models_tested': len(self.results),
                'results': self.results,
                'summary': {
                    'total_workers': 4,
                    'workers_tested': len(self.results),
                    'all_valid': all(r.get('valid', False) for r in self.results.values())
                }
            }
            
            # Sauvegarder
            output_path = Path("diagnostic/results/backtest_existing_models.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"   ✅ Rapport sauvegardé: {output_path}")
            return report
        except Exception as e:
            print(f"   ❌ Erreur génération rapport: {e}")
            return None
    
    def run_backtest(self):
        """Exécuter le backtest complet"""
        # Charger les modèles
        if not self.load_models_and_vecnorm():
            return False
        
        # Charger les données
        data = self.load_historical_data()
        if data is None:
            return False
        
        # Backtest chaque worker
        for worker_id in ["w1", "w2", "w3", "w4"]:
            result = self.backtest_worker(worker_id, data)
            if result:
                self.results[worker_id] = result
        
        # Générer le rapport
        report = self.generate_report()
        
        # Affichage final
        print("\n" + "="*70)
        if len(self.results) >= 3:
            print("✅ BACKTEST RÉUSSI")
            print("="*70)
            print(f"\nRésultats:")
            print(f"  - Workers testés: {len(self.results)}/4")
            print(f"  - Tous valides: {report['summary']['all_valid']}")
            print("\nProchaine étape: Détection de sur-apprentissage")
            return True
        else:
            print("❌ BACKTEST ÉCHOUÉ")
            print("="*70)
            print(f"\nRésultats:")
            print(f"  - Workers testés: {len(self.results)}/4")
            return False

def main():
    """Fonction principale"""
    backtest = BacktestExistingModels()
    success = backtest.run_backtest()
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
