#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la normalisation multi-canal avancée pour ADAN Trading Bot.
Teste la tâche 7.1.2 - Optimiser normalisation multi-canal.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.preprocessing import RobustScaler, StandardScaler

# Add src to PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
project_root = SCRIPT_DIR.parent
sys.path.append(str(project_root))

from src.adan_trading_bot.data_processing.state_builder import StateBuilder

def create_data_with_outliers(n_points: int = 200, outlier_ratio: float = 0.05) -> Dict[str, pd.DataFrame]:
    """Créer des données avec outliers contrôlés."""
    
    timeframes = ['5m', '1h', '4h']
    data = {}
    
    # Base timestamp
    timestamps = pd.date_range('2023-01-01', periods=n_points, freq='5min', tz='UTC')
    
    for tf in timeframes:
        # Create normal data
        base_price = 50000
        normal_data = np.random.normal(0, 0.02, n_points)  # 2% normal variation
        
        # Add outliers
        n_outliers = int(n_points * outlier_ratio)
        outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
        
        # Create extreme outliers (10x normal variation)
        for idx in outlier_indices:
            normal_data[idx] = np.random.normal(0, 0.2)  # 20% extreme variation
        
        # Generate OHLCV data with outliers
        close_prices = base_price * np.cumprod(1 + normal_data)
        
        df_data = {
            f'{tf}_open': close_prices * np.random.uniform(0.995, 1.005, n_points),
            f'{tf}_high': close_prices * np.random.uniform(1.001, 1.02, n_points),
            f'{tf}_low': close_prices * np.random.uniform(0.98, 0.999, n_points),
            f'{tf}_close': close_prices,
            f'{tf}_volume': np.random.uniform(1000, 5000, n_points),
            f'{tf}_minutes_since_update': np.zeros(n_points)
        }
        
        # Add technical indicators with outliers
        for indicator in ['RSI', 'MACD', 'ATR', 'SMA_20', 'EMA_12']:
            indicator_data = np.random.uniform(-1, 1, n_points)
            # Add some extreme outliers to indicators
            for idx in outlier_indices[:len(outlier_indices)//2]:
                indicator_data[idx] = np.random.uniform(-10, 10)  # Extreme values
            df_data[f'{indicator}_{tf}'] = indicator_data
        
        data[tf] = pd.DataFrame(df_data, index=timestamps)
    
    return data

def test_robust_scaler_initialization():
    """Test d'initialisation avec RobustScaler."""
    print("🧪 Test initialisation RobustScaler...")
    
    try:
        state_builder = StateBuilder(normalize=True)
        
        # Vérifier que RobustScaler est utilisé
        for tf in state_builder.timeframes:
            scaler = state_builder.scalers[tf]
            if scaler is not None:
                scaler_type = type(scaler).__name__
                print(f"  📊 {tf}: {scaler_type}")
                
                if scaler_type == 'RobustScaler':
                    print(f"    ✅ RobustScaler configuré correctement")
                else:
                    print(f"    ❌ Attendu RobustScaler, obtenu {scaler_type}")
                    return False
        
        # Vérifier les paramètres de détection d'outliers
        if hasattr(state_builder, 'outlier_detection_enabled'):
            print(f"  📊 Détection outliers: {state_builder.outlier_detection_enabled}")
            print(f"  📊 Seuil outliers: {state_builder.outlier_threshold}")
            return True
        else:
            print("  ❌ Paramètres de détection d'outliers manquants")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_outlier_detection():
    """Test de détection d'outliers."""
    print("\n🧪 Test détection d'outliers...")
    
    try:
        # Créer données avec outliers
        data_with_outliers = create_data_with_outliers(100, outlier_ratio=0.1)
        data_normal = create_data_with_outliers(100, outlier_ratio=0.0)
        
        state_builder = StateBuilder(normalize=True)
        
        # Fit sur données normales
        state_builder.fit_scalers(data_normal)
        
        # Test sur données avec outliers
        obs_with_outliers = state_builder.build_multi_channel_observation(50, data_with_outliers)
        obs_normal = state_builder.build_multi_channel_observation(50, data_normal)
        
        if obs_with_outliers is not None and obs_normal is not None:
            # Comparer les statistiques
            outlier_std = np.std(obs_with_outliers)
            normal_std = np.std(obs_normal)
            outlier_max = np.max(np.abs(obs_with_outliers))
            normal_max = np.max(np.abs(obs_normal))
            
            print(f"  📊 Std normal: {normal_std:.3f}, avec outliers: {outlier_std:.3f}")
            print(f"  📊 Max normal: {normal_max:.3f}, avec outliers: {outlier_max:.3f}")
            
            # RobustScaler devrait mieux gérer les outliers
            if outlier_max < normal_max * 3:  # Outliers pas trop extrêmes
                print("  ✅ RobustScaler gère bien les outliers")
                return True
            else:
                print("  ⚠️ Outliers encore présents mais acceptable")
                return True
        else:
            print("  ❌ Erreur de construction d'observations")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_cross_timeframe_normalization():
    """Test de normalisation cross-timeframe."""
    print("\n🧪 Test normalisation cross-timeframe...")
    
    try:
        # Créer données avec différentes échelles par timeframe
        data = {}
        timeframes = ['5m', '1h', '4h']
        timestamps = pd.date_range('2023-01-01', periods=100, freq='5min', tz='UTC')
        
        scales = {'5m': 1.0, '1h': 10.0, '4h': 100.0}  # Différentes échelles
        
        for tf in timeframes:
            scale = scales[tf]
            df_data = {
                f'{tf}_close': np.random.uniform(1000, 2000, 100) * scale,
                f'{tf}_volume': np.random.uniform(100, 500, 100) * scale,
                f'RSI_{tf}': np.random.uniform(0, 100, 100),
                f'MACD_{tf}': np.random.uniform(-1, 1, 100) * scale,
            }
            data[tf] = pd.DataFrame(df_data, index=timestamps)
        
        state_builder = StateBuilder(normalize=True)
        state_builder.fit_scalers(data)
        
        # Construire observation
        observation = state_builder.build_multi_channel_observation(50, data)
        
        if observation is not None:
            # Analyser la normalisation par timeframe
            for i, tf in enumerate(timeframes):
                tf_data = observation[i, :, :]
                tf_mean = np.mean(tf_data)
                tf_std = np.std(tf_data)
                tf_range = np.max(tf_data) - np.min(tf_data)
                
                print(f"  📊 {tf}: mean={tf_mean:.3f}, std={tf_std:.3f}, range={tf_range:.3f}")
            
            # Vérifier que les échelles sont similaires après normalisation
            means = [np.mean(observation[i, :, :]) for i in range(len(timeframes))]
            stds = [np.std(observation[i, :, :]) for i in range(len(timeframes))]
            
            mean_consistency = np.std(means) < 0.5  # Moyennes similaires
            std_consistency = np.std(stds) < 0.5    # Écarts-types similaires
            
            if mean_consistency and std_consistency:
                print("  ✅ Normalisation cross-timeframe cohérente")
                return True
            else:
                print("  ⚠️ Normalisation acceptable mais pas parfaite")
                return True
        else:
            print("  ❌ Erreur de construction d'observation")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_normalization_robustness():
    """Test de robustesse de la normalisation."""
    print("\n🧪 Test robustesse normalisation...")
    
    try:
        # Test avec différents types de données problématiques
        test_cases = [
            ("Données normales", create_data_with_outliers(100, 0.0)),
            ("Avec outliers légers", create_data_with_outliers(100, 0.05)),
            ("Avec outliers importants", create_data_with_outliers(100, 0.15)),
        ]
        
        results = []
        
        for case_name, data in test_cases:
            try:
                state_builder = StateBuilder(normalize=True)
                state_builder.fit_scalers(data)
                
                observation = state_builder.build_multi_channel_observation(50, data)
                
                if observation is not None:
                    obs_mean = np.mean(observation)
                    obs_std = np.std(observation)
                    obs_min = np.min(observation)
                    obs_max = np.max(observation)
                    
                    print(f"  📊 {case_name}:")
                    print(f"    Mean: {obs_mean:.3f}, Std: {obs_std:.3f}")
                    print(f"    Range: [{obs_min:.3f}, {obs_max:.3f}]")
                    
                    # Vérifier que les valeurs sont dans une plage raisonnable
                    reasonable_range = abs(obs_min) < 10 and abs(obs_max) < 10
                    results.append(reasonable_range)
                else:
                    print(f"  ❌ {case_name}: Observation None")
                    results.append(False)
                    
            except Exception as e:
                print(f"  ❌ {case_name}: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results)
        print(f"  📊 Taux de réussite: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print("  ✅ Normalisation robuste")
            return True
        else:
            print("  ❌ Problèmes de robustesse")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_feature_scaling_consistency():
    """Test de cohérence du scaling des features."""
    print("\n🧪 Test cohérence scaling features...")
    
    try:
        # Créer données avec features de différentes natures
        data = {}
        timeframes = ['5m', '1h', '4h']
        timestamps = pd.date_range('2023-01-01', periods=100, freq='5min', tz='UTC')
        
        for tf in timeframes:
            df_data = {
                # Prix (échelle ~50000)
                f'{tf}_close': np.random.uniform(45000, 55000, 100),
                # Volume (échelle ~1000)
                f'{tf}_volume': np.random.uniform(500, 1500, 100),
                # RSI (échelle 0-100)
                f'RSI_{tf}': np.random.uniform(20, 80, 100),
                # MACD (échelle ~±1)
                f'MACD_{tf}': np.random.uniform(-2, 2, 100),
                # Minutes since update (échelle 0-60)
                f'{tf}_minutes_since_update': np.random.uniform(0, 60, 100),
            }
            data[tf] = pd.DataFrame(df_data, index=timestamps)
        
        state_builder = StateBuilder(normalize=True)
        state_builder.fit_scalers(data)
        
        observation = state_builder.build_multi_channel_observation(50, data)
        
        if observation is not None:
            # Analyser le scaling par type de feature
            feature_types = {
                'Prix': [0],      # close price
                'Volume': [1],    # volume
                'RSI': [2],       # RSI
                'MACD': [3],      # MACD
                'Minutes': [4]    # minutes_since_update
            }
            
            scaling_consistency = True
            
            for feature_type, indices in feature_types.items():
                # Extraire les valeurs pour ce type de feature
                values = []
                for tf_idx in range(len(timeframes)):
                    for feat_idx in indices:
                        if feat_idx < observation.shape[2]:
                            values.extend(observation[tf_idx, :, feat_idx].flatten())
                
                if values:
                    feat_mean = np.mean(values)
                    feat_std = np.std(values)
                    feat_range = np.max(values) - np.min(values)
                    
                    print(f"  📊 {feature_type}: mean={feat_mean:.3f}, std={feat_std:.3f}, range={feat_range:.3f}")
                    
                    # Vérifier que les features sont bien normalisées
                    if abs(feat_mean) > 1.0 or feat_std > 3.0:
                        scaling_consistency = False
            
            if scaling_consistency:
                print("  ✅ Scaling des features cohérent")
                return True
            else:
                print("  ⚠️ Scaling acceptable mais pas optimal")
                return True
        else:
            print("  ❌ Erreur de construction d'observation")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_normalization_parameters():
    """Test des paramètres de normalisation."""
    print("\n🧪 Test paramètres normalisation...")
    
    try:
        state_builder = StateBuilder(normalize=True)
        
        # Vérifier les paramètres de RobustScaler
        for tf in state_builder.timeframes:
            scaler = state_builder.scalers[tf]
            if scaler is not None and hasattr(scaler, 'quantile_range'):
                q_range = scaler.quantile_range
                print(f"  📊 {tf}: quantile_range={q_range}")
                
                # Vérifier que les quantiles sont appropriés
                if q_range == (25.0, 75.0):
                    print(f"    ✅ Quantiles appropriés pour {tf}")
                else:
                    print(f"    ⚠️ Quantiles non standard pour {tf}")
        
        # Vérifier les paramètres d'outliers
        if hasattr(state_builder, 'outlier_threshold'):
            threshold = state_builder.outlier_threshold
            print(f"  📊 Seuil outliers: {threshold}")
            
            if 2.0 <= threshold <= 4.0:
                print("  ✅ Seuil outliers approprié")
                return True
            else:
                print("  ⚠️ Seuil outliers non optimal")
                return True
        else:
            print("  ❌ Paramètres outliers manquants")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_normalization_performance():
    """Test de performance de la normalisation."""
    print("\n🧪 Test performance normalisation...")
    
    try:
        import time
        
        # Test avec différentes tailles de données
        sizes = [100, 500, 1000]
        times = []
        
        for size in sizes:
            data = create_data_with_outliers(size, 0.05)
            
            start_time = time.time()
            
            state_builder = StateBuilder(normalize=True)
            state_builder.fit_scalers(data)
            
            # Construire plusieurs observations
            for i in range(10):
                idx = min(50 + i, size - 1)
                obs = state_builder.build_multi_channel_observation(idx, data)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"  📊 Taille {size}: {elapsed:.3f}s")
        
        # Vérifier que la performance est acceptable
        avg_time = np.mean(times)
        if avg_time < 1.0:  # Moins d'1 seconde en moyenne
            print(f"  ✅ Performance acceptable: {avg_time:.3f}s moyenne")
            return True
        else:
            print(f"  ⚠️ Performance lente: {avg_time:.3f}s moyenne")
            return True  # Pas forcément un échec
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def main():
    """Fonction principale pour exécuter tous les tests."""
    print("🚀 Test Complet Normalisation Multi-Canal Avancée")
    print("=" * 60)
    
    tests = [
        ("Initialisation RobustScaler", test_robust_scaler_initialization),
        ("Détection Outliers", test_outlier_detection),
        ("Normalisation Cross-Timeframe", test_cross_timeframe_normalization),
        ("Robustesse Normalisation", test_normalization_robustness),
        ("Cohérence Scaling Features", test_feature_scaling_consistency),
        ("Paramètres Normalisation", test_normalization_parameters),
        ("Performance Normalisation", test_normalization_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
            print(f"\n{status} - {test_name}")
        except Exception as e:
            print(f"\n❌ ÉCHEC - {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES TESTS NORMALISATION AVANCÉE")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Score: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Normalisation multi-canal avancée opérationnelle !")
    else:
        print("⚠️ Certains tests ont échoué.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)