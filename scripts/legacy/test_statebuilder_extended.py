#!/usr/bin/env python3
"""
Script de test pour valider l'adaptation du StateBuilder aux 22+ indicateurs techniques.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.data_processing.feature_engineer import AdvancedIndicatorCalculator

def create_sample_data_with_indicators(n_points=300):
    """Créer des données de test avec indicateurs techniques."""
    np.random.seed(42)
    
    # Générer des prix réalistes
    base_price = 50000
    returns = np.random.normal(0, 0.02, n_points)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    # Créer OHLCV de base
    base_data = {
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    }
    
    df = pd.DataFrame(base_data)
    
    # Ajouter des indicateurs techniques avec AdvancedIndicatorCalculator
    calculator = AdvancedIndicatorCalculator()
    
    # Calculer les indicateurs pour chaque timeframe
    timeframes = ['5m', '1h', '4h']
    data_with_indicators = {}
    
    for tf in timeframes:
        # Calculer les indicateurs
        df_with_indicators, indicators = calculator.calculate_all_indicators(df.copy(), tf)
        
        # Créer les colonnes avec préfixes de timeframe
        tf_data = {}
        
        # Colonnes de base avec préfixe
        for col in ['open', 'high', 'low', 'close', 'volume']:
            tf_data[f'{tf}_{col}'] = df_with_indicators[col]
        
        # Ajouter minutes_since_update
        tf_data[f'{tf}_minutes_since_update'] = np.random.randint(0, 60, n_points)
        
        # Ajouter tous les indicateurs calculés
        for indicator in indicators:
            if indicator in df_with_indicators.columns:
                tf_data[indicator] = df_with_indicators[indicator]
        
        data_with_indicators[tf] = pd.DataFrame(tf_data)
    
    return data_with_indicators

def test_extended_features_config():
    """Test de la configuration étendue des features."""
    print("🧪 Test Configuration Étendue des Features")
    print("=" * 50)
    
    # Créer StateBuilder avec configuration étendue
    state_builder = StateBuilder(window_size=50)
    
    # Vérifier la configuration générée
    config = state_builder.features_config
    
    print(f"📊 Timeframes configurés: {list(config.keys())}")
    
    total_features = 0
    for tf, features in config.items():
        print(f"  {tf}: {len(features)} features")
        print(f"    Exemples: {features[:5]}...")
        total_features += len(features)
    
    print(f"📈 Total features: {total_features}")
    print(f"📊 Observation shape: {state_builder.get_observation_shape()}")
    
    # Vérifier que les indicateurs attendus sont présents
    expected_indicators = ['RSI', 'MACD', 'ATR', 'SMA_20', 'BB_Upper', 'OBV', 'VWAP']
    
    for tf in config.keys():
        tf_features = config[tf]
        found_indicators = []
        
        for indicator in expected_indicators:
            matching_features = [f for f in tf_features if indicator in f]
            if matching_features:
                found_indicators.extend(matching_features)
        
        print(f"  {tf} - Indicateurs trouvés: {len(found_indicators)}")
        print(f"    Exemples: {found_indicators[:3]}...")
    
    return len(config) == 3 and total_features > 60  # Au moins 20 features par timeframe

def test_auto_detect_features():
    """Test de la détection automatique des features."""
    print("\n🔍 Test Détection Automatique des Features")
    print("=" * 50)
    
    # Créer des données avec indicateurs
    data = create_sample_data_with_indicators(200)
    
    # Créer StateBuilder
    state_builder = StateBuilder(window_size=50)
    
    # Détecter automatiquement les features
    detected_config = state_builder.auto_detect_features(data)
    
    print(f"📊 Features détectées par timeframe:")
    
    total_detected = 0
    for tf, features in detected_config.items():
        print(f"  {tf}: {len(features)} features détectées")
        print(f"    Exemples: {features[:5]}...")
        total_detected += len(features)
    
    print(f"📈 Total features détectées: {total_detected}")
    
    # Mettre à jour la configuration
    state_builder.update_features_config(detected_config)
    print(f"📊 Nouvelle observation shape: {state_builder.get_observation_shape()}")
    
    return total_detected > 30  # Au moins 10 features par timeframe

def test_feature_importance_analysis():
    """Test de l'analyse d'importance des features."""
    print("\n📊 Test Analyse d'Importance des Features")
    print("=" * 50)
    
    # Créer des données avec indicateurs
    data = create_sample_data_with_indicators(200)
    
    # Créer StateBuilder et détecter les features
    state_builder = StateBuilder(window_size=50)
    detected_config = state_builder.auto_detect_features(data)
    state_builder.update_features_config(detected_config)
    
    # Analyser l'importance des features
    importance_analysis = state_builder.get_feature_importance_analysis(data)
    
    print(f"📊 Analyse d'importance par timeframe:")
    
    for tf, importance_scores in importance_analysis.items():
        if not importance_scores:
            continue
            
        # Trier par importance décroissante
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  {tf} - Top 5 features les plus importantes:")
        for i, (feature, score) in enumerate(sorted_features[:5]):
            print(f"    {i+1}. {feature}: {score:.4f}")
        
        print(f"  {tf} - Features les moins importantes:")
        for i, (feature, score) in enumerate(sorted_features[-3:]):
            print(f"    {feature}: {score:.4f}")
    
    return len(importance_analysis) == 3

def test_observation_building():
    """Test de la construction d'observations avec les nouvelles features."""
    print("\n🏗️ Test Construction d'Observations")
    print("=" * 50)
    
    # Créer des données avec indicateurs
    data = create_sample_data_with_indicators(200)
    
    # Créer StateBuilder
    state_builder = StateBuilder(window_size=50)
    detected_config = state_builder.auto_detect_features(data)
    state_builder.update_features_config(detected_config)

    # --- Dimension Validation ---
    print("📏 Validating state dimension...")
    try:
        state_builder.validate_dimension(data)
        print("   ✅ State dimension validation successful.")
    except ValueError as e:
        print(f"   ❌ State dimension validation failed: {e}")
        raise e
    # --- End Validation ---
    
    # Construire une observation
    current_idx = 100
    
    # Test observation par timeframe
    observations = state_builder.build_observation(current_idx, data)
    
    print(f"📊 Observations construites:")
    for tf, obs in observations.items():
        print(f"  {tf}: shape {obs.shape}")
        print(f"    Min: {obs.min():.4f}, Max: {obs.max():.4f}")
        print(f"    NaN count: {np.isnan(obs).sum()}")
    
    # Test observation multi-canal
    multi_channel_obs = state_builder.build_multi_channel_observation(current_idx, data)
    
    if multi_channel_obs is not None:
        print(f"\n📈 Observation multi-canal:")
        print(f"  Shape: {multi_channel_obs.shape}")
        print(f"  Min: {multi_channel_obs.min():.4f}, Max: {multi_channel_obs.max():.4f}")
        print(f"  NaN count: {np.isnan(multi_channel_obs).sum()}")
        
        # Valider l'observation
        is_valid = state_builder.validate_observation(multi_channel_obs)
        print(f"  Validation: {'✅ Valide' if is_valid else '❌ Invalide'}")
        
        return is_valid and multi_channel_obs.shape[0] == 3
    
    return False

def test_scalers_with_extended_features():
    """Test des scalers avec les features étendues."""
    print("\n⚖️ Test Scalers avec Features Étendues")
    print("=" * 50)
    
    # Créer des données avec indicateurs
    data = create_sample_data_with_indicators(200)
    
    # Créer StateBuilder avec normalisation
    state_builder = StateBuilder(window_size=50, normalize=True)
    detected_config = state_builder.auto_detect_features(data)
    state_builder.update_features_config(detected_config)
    
    # Ajuster les scalers
    state_builder.fit_scalers(data)
    
    # Construire une observation normalisée
    current_idx = 100
    observations = state_builder.build_observation(current_idx, data)
    
    print(f"📊 Observations normalisées:")
    for tf, obs in observations.items():
        print(f"  {tf}: shape {obs.shape}")
        print(f"    Mean: {obs.mean():.4f}, Std: {obs.std():.4f}")
        print(f"    Min: {obs.min():.4f}, Max: {obs.max():.4f}")
    
    # Vérifier les statistiques de normalisation
    norm_stats = state_builder.get_normalization_stats()
    
    print(f"\n📈 Statistiques de normalisation:")
    for tf, stats in norm_stats.items():
        if 'mean' in stats:
            print(f"  {tf}: Mean shape {stats['mean'].shape}, Scale shape {stats['scale'].shape}")
    
    # Vérifier que les données sont bien normalisées (moyenne proche de 0, std proche de 1)
    all_normalized = True
    for tf, obs in observations.items():
        mean_abs = abs(obs.mean())
        if mean_abs > 0.5:  # Tolérance pour la moyenne
            all_normalized = False
            print(f"  ⚠️ {tf}: Moyenne trop éloignée de 0: {mean_abs:.4f}")
    
    return all_normalized

def main():
    """Fonction principale de test."""
    print("🚀 Test Complet StateBuilder Étendu")
    print("=" * 60)
    
    tests = [
        ("Configuration étendue", test_extended_features_config),
        ("Détection automatique", test_auto_detect_features),
        ("Analyse d'importance", test_feature_importance_analysis),
        ("Construction observations", test_observation_building),
        ("Scalers étendus", test_scalers_with_extended_features)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
            print(f"\n{status} - {test_name}")
        except Exception as e:
            print(f"\n❌ ÉCHEC - {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES TESTS STATEBUILDER")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Score: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 StateBuilder étendu opérationnel pour 22+ indicateurs !")
        return True
    else:
        print("⚠️ Certains tests ont échoué.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)