#!/usr/bin/env python3
"""
Diagnostic du pipeline d'observation - Vérifier que les workers reçoivent les bonnes données
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO

def diagnose_worker_dimensions():
    """Vérifier les dimensions attendues par chaque worker"""
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC 1: DIMENSIONS DES WORKERS")
    print("="*70)
    
    workers_info = {}
    for i in range(1, 5):
        model_path = f"/mnt/new_data/t10_training/checkpoints/w{i}/w{i}_model_350000_steps.zip"
        try:
            model = PPO.load(model_path)
            obs_shape = model.observation_space.shape
            workers_info[f"w{i}"] = {
                "obs_shape": obs_shape,
                "obs_dim": obs_shape[0] if obs_shape else None,
                "action_shape": model.action_space.shape
            }
            print(f"✅ w{i}:")
            print(f"   - Observation shape: {obs_shape}")
            print(f"   - Action shape: {model.action_space.shape}")
        except Exception as e:
            print(f"❌ w{i}: Erreur - {e}")
    
    return workers_info

def check_production_scalers():
    """Vérifier les scalers de production"""
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC 2: SCALERS DE PRODUCTION")
    print("="*70)
    
    prod_scalers_dir = Path("/mnt/new_data/t10_training/phase2_results/prod_scalers")
    
    if not prod_scalers_dir.exists():
        print(f"❌ Répertoire prod_scalers manquant: {prod_scalers_dir}")
        return None
    
    import joblib
    scalers_info = {}
    
    for scaler_file in sorted(prod_scalers_dir.glob("*.pkl")):
        try:
            scaler = joblib.load(scaler_file)
            if hasattr(scaler, 'n_features_in_'):
                scalers_info[scaler_file.name] = scaler.n_features_in_
                print(f"✅ {scaler_file.name}: {scaler.n_features_in_} features")
            else:
                print(f"⚠️  {scaler_file.name}: Pas d'attribut n_features_in_")
        except Exception as e:
            print(f"❌ {scaler_file.name}: Erreur - {e}")
    
    return scalers_info

def check_historical_data():
    """Vérifier les données historiques préchargées"""
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC 3: DONNÉES HISTORIQUES")
    print("="*70)
    
    data_dir = Path("historical_data")
    data_info = {}
    
    for tf in ['5m', '1h', '4h']:
        file_path = data_dir / f"BTC_USDT_{tf}_data.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                data_info[tf] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                }
                print(f"✅ {tf}: {len(df)} lignes, {len(df.columns)} colonnes")
                print(f"   Colonnes: {', '.join(df.columns[:5])}...")
            except Exception as e:
                print(f"❌ {tf}: Erreur - {e}")
        else:
            print(f"❌ {tf}: Fichier manquant - {file_path}")
    
    return data_info

def check_paper_trading_state():
    """Vérifier l'état du paper trading"""
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC 4: ÉTAT DU PAPER TRADING")
    print("="*70)
    
    state_file = Path("/mnt/new_data/t10_training/phase2_results/paper_trading_state.json")
    
    if not state_file.exists():
        print(f"❌ Fichier d'état manquant: {state_file}")
        return None
    
    try:
        with open(state_file) as f:
            state = json.load(f)
        
        print(f"✅ État du portfolio:")
        portfolio = state.get('portfolio', {})
        print(f"   - Balance: ${portfolio.get('balance', 0):.2f}")
        print(f"   - Equity: ${portfolio.get('equity', 0):.2f}")
        print(f"   - Positions ouvertes: {len(portfolio.get('positions', []))}")
        
        for pos in portfolio.get('positions', []):
            print(f"     • {pos['pair']}: {pos['side']} @ {pos['entry_price']:.2f}")
            print(f"       Current: {pos.get('current_price', pos['entry_price']):.2f}")
            print(f"       PnL: {pos.get('pnl_pct', 0):+.2f}%")
        
        return state
    except Exception as e:
        print(f"❌ Erreur lecture état: {e}")
        return None

def check_latest_prediction():
    """Vérifier la dernière prédiction"""
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC 5: DERNIÈRE PRÉDICTION")
    print("="*70)
    
    state_file = Path("/mnt/new_data/t10_training/phase2_results/paper_trading_state.json")
    
    try:
        with open(state_file) as f:
            state = json.load(f)
        
        prediction = state.get('latest_prediction', {})
        if prediction:
            print(f"✅ Dernière prédiction:")
            print(f"   - Signal: {prediction.get('signal', 'N/A')}")
            print(f"   - Confiance: {prediction.get('confidence', 0):.3f}")
            print(f"   - Worker votes:")
            for worker, conf in prediction.get('worker_votes', {}).items():
                print(f"     • {worker}: {conf:.3f}")
        else:
            print(f"⚠️  Aucune prédiction enregistrée")
        
        return prediction
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def main():
    print("\n" + "🔧 DIAGNOSTIC COMPLET DU PIPELINE D'OBSERVATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Exécuter tous les diagnostics
    workers = diagnose_worker_dimensions()
    scalers = check_production_scalers()
    data = check_historical_data()
    state = check_paper_trading_state()
    prediction = check_latest_prediction()
    
    # Résumé
    print("\n" + "="*70)
    print("📋 RÉSUMÉ DES PROBLÈMES DÉTECTÉS")
    print("="*70)
    
    issues = []
    
    # Vérifier les dimensions
    if workers:
        expected_dims = list(workers.values())[0]['obs_dim']
        print(f"\n✅ Tous les workers attendent: {expected_dims} dimensions")
    
    # Vérifier les scalers
    if scalers:
        print(f"\n✅ Scalers trouvés: {len(scalers)}")
        for name, dim in scalers.items():
            print(f"   - {name}: {dim} features")
    else:
        issues.append("❌ Aucun scaler de production trouvé")
    
    # Vérifier les données
    if data:
        print(f"\n✅ Données historiques disponibles: {len(data)} timeframes")
    else:
        issues.append("❌ Données historiques manquantes")
    
    # Vérifier l'état
    if state:
        positions = len(state.get('portfolio', {}).get('positions', []))
        print(f"\n✅ État du système: {positions} position(s) ouverte(s)")
    else:
        issues.append("❌ État du système inaccessible")
    
    # Afficher les problèmes
    if issues:
        print("\n" + "="*70)
        print("🚨 PROBLÈMES À RÉSOUDRE")
        print("="*70)
        for issue in issues:
            print(issue)
    else:
        print("\n✅ Aucun problème détecté - Pipeline OK")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
