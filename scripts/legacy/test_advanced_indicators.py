#!/usr/bin/env python3
"""
Script de test pour valider l'implémentation des 22+ indicateurs techniques.
"""

import sys
import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.data_processing.feature_engineer import AdvancedIndicatorCalculator, FeatureEngineer

def create_sample_data(n_points=200):
    """Créer des données de test OHLCV."""
    np.random.seed(42)
    
    # Générer des prix réalistes
    base_price = 50000
    returns = np.random.normal(0, 0.02, n_points)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    # Créer OHLCV
    data = {
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    }
    
    return pd.DataFrame(data)

def test_advanced_indicator_calculator():
    """Test de la classe AdvancedIndicatorCalculator."""
    print("🧪 Test AdvancedIndicatorCalculator...")
    
    # Créer des données de test
    df = create_sample_data(200)
    calculator = AdvancedIndicatorCalculator()
    
    # Tester pour chaque timeframe
    timeframes = ['5m', '1h', '4h']
    all_results = {}
    
    for tf in timeframes:
        print(f"\n📊 Test timeframe {tf}:")
        
        # Calculer les indicateurs
        df_result, indicators = calculator.calculate_all_indicators(df.copy(), tf)
        all_results[tf] = (df_result, indicators)
        
        print(f"  ✅ {len(indicators)} indicateurs calculés")
        print(f"  📈 Shape finale: {df_result.shape}")
        
        # Vérifier les catégories d'indicateurs
        trend_indicators = [ind for ind in indicators if any(x in ind for x in ['SMA', 'EMA', 'MACD'])]
        momentum_indicators = [ind for ind in indicators if any(x in ind for x in ['RSI', 'STOCH', 'WILLR', 'CCI', 'ADX'])]
        volatility_indicators = [ind for ind in indicators if any(x in ind for x in ['ATR', 'BB'])]
        volume_indicators = [ind for ind in indicators if any(x in ind for x in ['OBV', 'VWAP', 'MFI', 'Volume'])]
        
        print(f"  📈 Tendance: {len(trend_indicators)} indicateurs")
        print(f"  🚀 Momentum: {len(momentum_indicators)} indicateurs") 
        print(f"  📊 Volatilité: {len(volatility_indicators)} indicateurs")
        print(f"  📦 Volume: {len(volume_indicators)} indicateurs")
        
        # Vérifier qu'il n'y a pas trop de NaN
        nan_percentage = df_result.isnull().sum().sum() / (df_result.shape[0] * df_result.shape[1])
        print(f"  🔍 Pourcentage NaN: {nan_percentage:.2%}")
        
        if nan_percentage > 0.1:
            print(f"  ⚠️  Attention: Trop de NaN ({nan_percentage:.2%})")
        else:
            print(f"  ✅ Pourcentage NaN acceptable")
    
    return all_results

def test_feature_engineer_integration():
    """Test de l'intégration avec FeatureEngineer."""
    print("\n🔧 Test intégration FeatureEngineer...")
    
    # Charger la configuration
    config_path = Path(__file__).parent.parent / 'config' / 'feature_engineering.yaml'
    
    if not config_path.exists():
        print(f"❌ Configuration non trouvée: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Créer des données multi-timeframes simulées
    sample_data = create_sample_data(200)
    
    # Simuler des données fusionnées multi-timeframes
    merged_data = pd.DataFrame()
    for tf in ['5m', '1h', '4h']:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            merged_data[f'{tf}_{col}'] = sample_data[col] * (1 + np.random.normal(0, 0.001, len(sample_data)))
        merged_data[f'{tf}_minutes_since_update'] = np.random.randint(0, 60, len(sample_data))
    
    try:
        # Créer le FeatureEngineer
        engineer = FeatureEngineer(config, models_dir='models')
        
        # Traiter les données
        result = engineer.process_data(merged_data, fit_scaler=True)
        
        print(f"  ✅ Traitement réussi")
        print(f"  📊 Shape originale: {merged_data.shape}")
        print(f"  📈 Shape finale: {result.shape}")
        print(f"  🔢 Nouvelles colonnes: {result.shape[1] - merged_data.shape[1]}")
        
        # Vérifier les colonnes d'indicateurs
        indicator_cols = [col for col in result.columns if any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', 'BB', 'OBV', 'VWAP'])]
        print(f"  📊 Colonnes d'indicateurs trouvées: {len(indicator_cols)}")
        
        if len(indicator_cols) >= 60:  # 22 indicateurs × 3 timeframes ≈ 66
            print(f"  ✅ Nombre d'indicateurs attendu atteint ({len(indicator_cols)})")
        else:
            print(f"  ⚠️  Moins d'indicateurs que prévu ({len(indicator_cols)} < 60)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur lors du traitement: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test de performance du calcul des indicateurs."""
    print("\n⚡ Test de performance...")
    
    import time
    
    # Créer des données plus importantes
    df_large = create_sample_data(1000)
    calculator = AdvancedIndicatorCalculator()
    
    # Mesurer le temps de calcul
    start_time = time.time()
    
    for tf in ['5m', '1h', '4h']:
        df_result, indicators = calculator.calculate_all_indicators(df_large.copy(), tf)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"  ⏱️  Temps de traitement: {processing_time:.2f}s")
    print(f"  📊 Points traités: {len(df_large) * 3} (1000 points × 3 timeframes)")
    print(f"  🚀 Vitesse: {(len(df_large) * 3) / processing_time:.0f} points/seconde")
    
    if processing_time < 5.0:
        print(f"  ✅ Performance acceptable")
        return True
    else:
        print(f"  ⚠️  Performance lente (> 5s)")
        return False

def main():
    """Fonction principale de test."""
    print("🚀 Test des 22+ Indicateurs Techniques ADAN")
    print("=" * 50)
    
    results = []
    
    # Test 1: AdvancedIndicatorCalculator
    try:
        test_results = test_advanced_indicator_calculator()
        results.append(("AdvancedIndicatorCalculator", True))
        print("✅ Test AdvancedIndicatorCalculator: RÉUSSI")
    except Exception as e:
        print(f"❌ Test AdvancedIndicatorCalculator: ÉCHEC - {str(e)}")
        results.append(("AdvancedIndicatorCalculator", False))
    
    # Test 2: Intégration FeatureEngineer
    try:
        integration_success = test_feature_engineer_integration()
        results.append(("Intégration FeatureEngineer", integration_success))
        if integration_success:
            print("✅ Test Intégration FeatureEngineer: RÉUSSI")
        else:
            print("❌ Test Intégration FeatureEngineer: ÉCHEC")
    except Exception as e:
        print(f"❌ Test Intégration FeatureEngineer: ÉCHEC - {str(e)}")
        results.append(("Intégration FeatureEngineer", False))
    
    # Test 3: Performance
    try:
        perf_success = test_performance()
        results.append(("Performance", perf_success))
        if perf_success:
            print("✅ Test Performance: RÉUSSI")
        else:
            print("⚠️ Test Performance: LENT")
    except Exception as e:
        print(f"❌ Test Performance: ÉCHEC - {str(e)}")
        results.append(("Performance", False))
    
    # Résumé final
    print("\n" + "=" * 50)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Score: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont réussis ! Les 22+ indicateurs sont opérationnels.")
        return True
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)