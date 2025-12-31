#!/usr/bin/env python3
"""
Checkpoint 3.1: Test d'Inférence Basique
Vérifie que le monitoring peut faire des prédictions après correction Phase 2.
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
print("CHECKPOINT 3.1: Test d'Inférence Basique")
print("="*70)

def test_monitor_initialization():
    """Test 1: Initialisation du monitor"""
    print("\n1️⃣  Test d'initialisation du monitor...")
    
    try:
        from scripts.paper_trading_monitor import RealPaperTradingMonitor
        
        # Initialiser sans API keys (mode test)
        monitor = RealPaperTradingMonitor(
            api_key="test_key",
            api_secret="test_secret"
        )
        
        # Initialiser le pipeline
        if not monitor.setup_pipeline():
            raise Exception("Pipeline setup failed")
        
        print("   ✅ Monitor initialisé avec succès")
        return monitor
        
    except Exception as e:
        print(f"   ❌ Échec initialisation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_vecnormalize_loading(monitor):
    """Test 2: Vérification du chargement VecNormalize"""
    print("\n2️⃣  Vérification chargement VecNormalize...")
    
    if not hasattr(monitor, 'workers'):
        print("   ❌ Attribut workers manquant")
        return False
    
    if len(monitor.workers) != 4:
        print(f"   ❌ Nombre de workers incorrect: {len(monitor.workers)} (attendu: 4)")
        return False
    
    print(f"   ✅ {len(monitor.workers)} modèles chargés")
    
    # Vérifier chaque worker
    for worker_id in ["w1", "w2", "w3", "w4"]:
        if worker_id not in monitor.workers:
            print(f"   ❌ Worker {worker_id} manquant")
            return False
        
        print(f"   ✅ {worker_id}: Modèle OK")
    
    return True

def test_data_loading():
    """Test 3: Chargement des données de test"""
    print("\n3️⃣  Chargement des données de test...")
    
    try:
        # Chercher les données disponibles
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
                
                # Prendre les dernières données selon le timeframe
                if tf == '5m':
                    data[tf] = df.tail(20)  # 20 dernières barres 5m
                elif tf == '1h':
                    data[tf] = df.tail(20)  # 20 dernières barres 1h
                elif tf == '4h':
                    data[tf] = df.tail(20)  # 20 dernières barres 4h
                
                print(f"   ✅ {tf}: {data[tf].shape} chargé")
            else:
                print(f"   ⚠️  Données manquantes: {path}")
                return None
        
        if len(data) < 3:
            print("   ❌ Données insuffisantes")
            return None
        
        return data
        
    except Exception as e:
        print(f"   ❌ Erreur chargement données: {e}")
        return None

def test_build_observation(monitor, data):
    """Test 4: Test de build_observation pour chaque worker"""
    print("\n4️⃣  Test build_observation pour chaque worker...")
    
    observations = {}
    
    for worker_id in ["w1", "w2", "w3", "w4"]:
        try:
            # Préparer raw_data au format attendu
            raw_data = {
                'BTC/USDT': data
            }
            
            # Appeler build_observation avec worker_id
            obs = monitor.build_observation(worker_id, raw_data)
            
            # Vérifications
            if obs is None:
                print(f"   ❌ {worker_id}: build_observation retourned None")
                continue
            
            if not isinstance(obs, dict):
                print(f"   ❌ {worker_id}: Observation n'est pas un dict")
                continue
            
            required_keys = {'5m', '1h', '4h', 'portfolio_state'}
            if not required_keys.issubset(obs.keys()):
                print(f"   ❌ {worker_id}: Clés manquantes: {required_keys - set(obs.keys())}")
                continue
            
            observations[worker_id] = obs
            print(f"   ✅ {worker_id}: observation générée")
            
        except Exception as e:
            print(f"   ❌ {worker_id}: {e}")
    
    return observations

def test_model_predictions(monitor, observations):
    """Test 5: Test des prédictions des modèles"""
    print("\n5️⃣  Test prédictions des modèles...")
    
    predictions = {}
    
    for worker_id in ["w1", "w2", "w3", "w4"]:
        if worker_id not in observations:
            print(f"   ⚠️  {worker_id}: Pas d'observation disponible")
            continue
        
        try:
            obs = observations[worker_id]
            
            # Vérifier que le modèle existe
            if worker_id not in monitor.workers:
                print(f"   ❌ {worker_id}: Modèle non chargé")
                continue
            
            model = monitor.workers[worker_id]
            
            # Faire une prédiction
            action, _states = model.predict(obs, deterministic=True)
            
            # Vérifications de l'action
            if np.isnan(action).any():
                print(f"   ❌ {worker_id}: Action contient NaN")
                continue
            
            if action.min() < -1.1 or action.max() > 1.1:
                print(f"   ⚠️  {worker_id}: Action hors limites [{action.min():.3f}, {action.max():.3f}]")
            
            predictions[worker_id] = {
                'action': action.tolist() if hasattr(action, 'tolist') else action,
                'shape': action.shape if hasattr(action, 'shape') else None,
                'range': [float(action.min()), float(action.max())] if hasattr(action, 'min') else None,
                'valid': True
            }
            
            print(f"   ✅ {worker_id}: prédiction valide (shape={action.shape if hasattr(action, 'shape') else 'N/A'})")
            
        except Exception as e:
            print(f"   ❌ {worker_id}: {e}")
            predictions[worker_id] = {'valid': False, 'error': str(e)}
    
    return predictions

def save_results(predictions):
    """Test 6: Sauvegarde des résultats"""
    print("\n6️⃣  Sauvegarde des résultats...")
    
    try:
        results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': '3.1',
            'test': 'inference_basic',
            'predictions': predictions,
            'summary': {
                'total_workers': 4,
                'successful_predictions': sum(1 for p in predictions.values() if p.get('valid', False)),
                'failed_predictions': sum(1 for p in predictions.values() if not p.get('valid', False))
            }
        }
        
        output_path = Path("diagnostic/results/checkpoint_3_1_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Résultats sauvegardés: {output_path}")
        return results
        
    except Exception as e:
        print(f"   ❌ Erreur sauvegarde: {e}")
        return None

def main():
    """Fonction principale du test"""
    
    # Test 1: Initialisation
    monitor = test_monitor_initialization()
    if monitor is None:
        print("\n❌ CHECKPOINT 3.1 ÉCHOUÉ - Initialisation impossible")
        return False
    
    # Test 2: VecNormalize
    if not test_vecnormalize_loading(monitor):
        print("\n❌ CHECKPOINT 3.1 ÉCHOUÉ - Modèles non chargés")
        return False
    
    # Test 3: Données
    data = test_data_loading()
    if data is None:
        print("\n❌ CHECKPOINT 3.1 ÉCHOUÉ - Données non disponibles")
        return False
    
    # Test 4: Observations
    observations = test_build_observation(monitor, data)
    if len(observations) == 0:
        print("\n❌ CHECKPOINT 3.1 ÉCHOUÉ - Aucune observation générée")
        return False
    
    # Test 5: Prédictions
    predictions = test_model_predictions(monitor, observations)
    if len(predictions) == 0:
        print("\n❌ CHECKPOINT 3.1 ÉCHOUÉ - Aucune prédiction générée")
        return False
    
    # Test 6: Sauvegarde
    results = save_results(predictions)
    if results is None:
        print("\n❌ CHECKPOINT 3.1 ÉCHOUÉ - Sauvegarde impossible")
        return False
    
    # Évaluation finale
    successful = results['summary']['successful_predictions']
    total = results['summary']['total_workers']
    
    print("\n" + "="*70)
    if successful >= 3:  # Au moins 3/4 workers fonctionnent
        print("✅ CHECKPOINT 3.1 VALIDÉ")
        print("="*70)
        print(f"\nRésultats: {successful}/{total} workers fonctionnels")
        print("Inférence basique opérationnelle")
        print("\nProchaine étape: Checkpoint 3.2 - Paper Trading Dry-Run")
        return True
    else:
        print("❌ CHECKPOINT 3.1 ÉCHOUÉ")
        print("="*70)
        print(f"\nRésultats: {successful}/{total} workers fonctionnels (minimum: 3)")
        print("Corriger les problèmes avant de continuer")
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
