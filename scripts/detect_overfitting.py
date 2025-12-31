#!/usr/bin/env python3
"""
Détection de Sur-Apprentissage
Compare performance in-sample vs out-of-sample pour détecter le sur-apprentissage
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
print("DÉTECTION DE SUR-APPRENTISSAGE")
print("="*70)

class OverfittingDetector:
    """Détecte le sur-apprentissage en comparant train vs test"""
    
    def __init__(self):
        self.results = {}
    
    def load_data_splits(self):
        """Charger les données en splits train/test"""
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
            
            # Diviser en train (50%) et test (50%)
            split_idx = {tf: len(data[tf]) // 2 for tf in data.keys()}
            
            data_train = {tf: data[tf].iloc[:split_idx[tf]] for tf in data.keys()}
            data_test = {tf: data[tf].iloc[split_idx[tf]:] for tf in data.keys()}
            
            print(f"\n   Train/Test Split:")
            for tf in data.keys():
                print(f"      {tf}: {len(data_train[tf])} train / {len(data_test[tf])} test")
            
            return data_train, data_test
        except Exception as e:
            print(f"   ❌ Erreur chargement: {e}")
            return None, None
    
    def calculate_metrics(self, actions_list):
        """Calculer les métriques de performance"""
        if len(actions_list) == 0:
            return None
        
        actions = np.array(actions_list)
        
        return {
            'num_actions': len(actions_list),
            'mean': float(actions.mean()),
            'std': float(actions.std()),
            'min': float(actions.min()),
            'max': float(actions.max()),
            'median': float(np.median(actions))
        }
    
    def test_worker_on_data(self, worker_id, data):
        """Tester un worker sur un ensemble de données"""
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
            
            # Collecter les actions
            actions_list = []
            num_iterations = min(len(data['5m']), 50)  # Limiter à 50 itérations
            
            for i in range(num_iterations):
                try:
                    raw_data = {
                        'BTC/USDT': {
                            '5m': data['5m'].iloc[max(0, i-20):i+1],
                            '1h': data['1h'].iloc[max(0, i-20):i+1],
                            '4h': data['4h'].iloc[max(0, i-20):i+1]
                        }
                    }
                    
                    obs = monitor.build_observation(worker_id, raw_data)
                    if obs is None:
                        continue
                    
                    action, _ = model.predict(obs, deterministic=True)
                    actions_list.append(float(action.mean()))
                except:
                    continue
            
            return self.calculate_metrics(actions_list)
        except Exception as e:
            print(f"   ❌ Erreur test {worker_id}: {e}")
            return None
    
    def detect_overfitting(self, data_train, data_test):
        """Détecter le sur-apprentissage pour chaque worker"""
        print("\n2️⃣  Détection du sur-apprentissage...")
        
        for worker_id in ["w1", "w2", "w3", "w4"]:
            print(f"\n   {worker_id}:")
            
            # Test sur données d'entraînement
            metrics_train = self.test_worker_on_data(worker_id, data_train)
            if metrics_train is None:
                print(f"      ⚠️  Impossible de tester sur données train")
                continue
            
            # Test sur données de test
            metrics_test = self.test_worker_on_data(worker_id, data_test)
            if metrics_test is None:
                print(f"      ⚠️  Impossible de tester sur données test")
                continue
            
            # Calculer la dégradation
            degradation = {
                'mean': abs(metrics_train['mean'] - metrics_test['mean']) / (abs(metrics_train['mean']) + 1e-6),
                'std': abs(metrics_train['std'] - metrics_test['std']) / (abs(metrics_train['std']) + 1e-6)
            }
            
            # Évaluer le sur-apprentissage
            if degradation['mean'] > 0.5:
                status = "SEVERE_OVERFITTING"
                symbol = "❌"
            elif degradation['mean'] > 0.3:
                status = "MODERATE_OVERFITTING"
                symbol = "⚠️"
            else:
                status = "OK"
                symbol = "✅"
            
            print(f"      {symbol} Status: {status}")
            print(f"         Train Mean: {metrics_train['mean']:.4f}")
            print(f"         Test Mean: {metrics_test['mean']:.4f}")
            print(f"         Dégradation: {degradation['mean']*100:.1f}%")
            
            self.results[worker_id] = {
                'status': status,
                'metrics_train': metrics_train,
                'metrics_test': metrics_test,
                'degradation': degradation
            }
    
    def generate_report(self):
        """Générer le rapport"""
        print("\n3️⃣  Génération du rapport...")
        try:
            # Compter les statuts
            severe = sum(1 for r in self.results.values() if r['status'] == 'SEVERE_OVERFITTING')
            moderate = sum(1 for r in self.results.values() if r['status'] == 'MODERATE_OVERFITTING')
            ok = sum(1 for r in self.results.values() if r['status'] == 'OK')
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'test': 'overfitting_detection',
                'results': self.results,
                'summary': {
                    'total_workers': len(self.results),
                    'severe_overfitting': severe,
                    'moderate_overfitting': moderate,
                    'ok': ok,
                    'recommendation': self._get_recommendation(severe, moderate, ok)
                }
            }
            
            # Sauvegarder
            output_path = Path("diagnostic/results/overfitting_detection.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"   ✅ Rapport sauvegardé: {output_path}")
            return report
        except Exception as e:
            print(f"   ❌ Erreur génération rapport: {e}")
            return None
    
    def _get_recommendation(self, severe, moderate, ok):
        """Obtenir une recommandation basée sur les résultats"""
        if severe > 0:
            return "RÉENTRAÎNER - Sur-apprentissage sévère détecté"
        elif moderate > 1:
            return "ATTENTION - Sur-apprentissage modéré, monitorer"
        else:
            return "OK - Pas de sur-apprentissage significatif"
    
    def run_detection(self):
        """Exécuter la détection complète"""
        # Charger les données
        data_train, data_test = self.load_data_splits()
        if data_train is None or data_test is None:
            return False
        
        # Détecter le sur-apprentissage
        self.detect_overfitting(data_train, data_test)
        
        # Générer le rapport
        report = self.generate_report()
        
        # Affichage final
        print("\n" + "="*70)
        if report:
            print("✅ DÉTECTION COMPLÉTÉE")
            print("="*70)
            print(f"\nRésumé:")
            print(f"  - OK: {report['summary']['ok']}/4")
            print(f"  - Modéré: {report['summary']['moderate_overfitting']}/4")
            print(f"  - Sévère: {report['summary']['severe_overfitting']}/4")
            print(f"\nRecommandation: {report['summary']['recommendation']}")
            return True
        else:
            print("❌ DÉTECTION ÉCHOUÉE")
            return False

def main():
    """Fonction principale"""
    detector = OverfittingDetector()
    success = detector.run_detection()
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
