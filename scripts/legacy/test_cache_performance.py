#!/usr/bin/env python3
"""
Script de test pour valider les performances du cache intelligent des indicateurs.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.data_processing.feature_engineer import AdvancedIndicatorCalculator

def create_sample_data(n_points=500):
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

def test_cache_performance():
    """Test des performances du cache intelligent."""
    print("🚀 Test Performance Cache Intelligent")
    print("=" * 50)
    
    # Créer des données de test
    df = create_sample_data(500)
    
    # Test avec cache activé
    print("\n📊 Test avec cache ACTIVÉ:")
    calculator_with_cache = AdvancedIndicatorCalculator(enable_cache=True, cache_size=100)
    
    # Premier calcul (cache miss)
    start_time = time.time()
    result1, indicators1 = calculator_with_cache.calculate_all_indicators(df.copy(), '1h')
    first_calc_time = time.time() - start_time
    
    # Deuxième calcul identique (cache hit)
    start_time = time.time()
    result2, indicators2 = calculator_with_cache.calculate_all_indicators(df.copy(), '1h')
    second_calc_time = time.time() - start_time
    
    # Statistiques du cache
    cache_stats = calculator_with_cache.get_cache_stats()
    
    print(f"  ⏱️  Premier calcul (cache miss): {first_calc_time:.4f}s")
    print(f"  ⏱️  Deuxième calcul (cache hit): {second_calc_time:.4f}s")
    print(f"  🚀 Accélération: {first_calc_time / second_calc_time:.1f}x")
    print(f"  📊 Cache hits: {cache_stats['cache_hits']}")
    print(f"  📊 Cache misses: {cache_stats['cache_misses']}")
    print(f"  📊 Hit rate: {cache_stats['hit_rate']:.1%}")
    
    # Test avec cache désactivé
    print("\n📊 Test avec cache DÉSACTIVÉ:")
    calculator_no_cache = AdvancedIndicatorCalculator(enable_cache=False)
    
    # Premier calcul
    start_time = time.time()
    result3, indicators3 = calculator_no_cache.calculate_all_indicators(df.copy(), '1h')
    no_cache_time1 = time.time() - start_time
    
    # Deuxième calcul
    start_time = time.time()
    result4, indicators4 = calculator_no_cache.calculate_all_indicators(df.copy(), '1h')
    no_cache_time2 = time.time() - start_time
    
    print(f"  ⏱️  Premier calcul: {no_cache_time1:.4f}s")
    print(f"  ⏱️  Deuxième calcul: {no_cache_time2:.4f}s")
    print(f"  📊 Différence: {abs(no_cache_time1 - no_cache_time2):.4f}s")
    
    # Comparaison globale
    print("\n📈 COMPARAISON GLOBALE:")
    print(f"  🚀 Gain avec cache: {no_cache_time2 / second_calc_time:.1f}x plus rapide")
    print(f"  💾 Efficacité cache: {(1 - second_calc_time / first_calc_time) * 100:.1f}% de réduction")
    
    # Vérifier que les résultats sont identiques
    results_identical = (
        len(indicators1) == len(indicators2) == len(indicators3) == len(indicators4) and
        np.allclose(result1.select_dtypes(include=[np.number]).fillna(0), 
                   result2.select_dtypes(include=[np.number]).fillna(0), rtol=1e-10)
    )
    
    print(f"  ✅ Résultats identiques: {'Oui' if results_identical else 'Non'}")
    
    return {
        'cache_acceleration': first_calc_time / second_calc_time,
        'cache_efficiency': (1 - second_calc_time / first_calc_time) * 100,
        'hit_rate': cache_stats['hit_rate'],
        'results_identical': results_identical
    }

def test_cache_with_different_data():
    """Test du cache avec différents jeux de données."""
    print("\n🔄 Test Cache avec Données Différentes")
    print("=" * 50)
    
    calculator = AdvancedIndicatorCalculator(enable_cache=True, cache_size=50)
    
    # Créer plusieurs jeux de données différents
    datasets = []
    for i in range(5):
        np.random.seed(i + 100)  # Différentes seeds pour données différentes
        datasets.append(create_sample_data(300))
    
    # Calculer les indicateurs pour chaque dataset
    calculation_times = []
    
    for i, df in enumerate(datasets):
        start_time = time.time()
        result, indicators = calculator.calculate_all_indicators(df, '5m')
        calc_time = time.time() - start_time
        calculation_times.append(calc_time)
        
        print(f"  Dataset {i+1}: {calc_time:.4f}s - {len(indicators)} indicateurs")
    
    # Recalculer le premier dataset (devrait être en cache)
    start_time = time.time()
    result, indicators = calculator.calculate_all_indicators(datasets[0], '5m')
    cached_time = time.time() - start_time
    
    print(f"  Dataset 1 (cached): {cached_time:.4f}s - {len(indicators)} indicateurs")
    
    # Statistiques finales
    cache_stats = calculator.get_cache_stats()
    print(f"\n📊 Statistiques finales:")
    print(f"  Cache size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Accélération sur recalcul: {calculation_times[0] / cached_time:.1f}x")

def test_cache_memory_usage():
    """Test de l'utilisation mémoire du cache."""
    print("\n💾 Test Utilisation Mémoire Cache")
    print("=" * 50)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Mesure mémoire initiale
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Créer calculateur avec grand cache
    calculator = AdvancedIndicatorCalculator(enable_cache=True, cache_size=100)
    
    # Remplir le cache avec de nombreux calculs
    for i in range(50):
        df = create_sample_data(200 + i * 10)  # Tailles variables
        calculator.calculate_all_indicators(df, f'test_{i}')
    
    # Mesure mémoire finale
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    cache_stats = calculator.get_cache_stats()
    
    print(f"  📊 Mémoire initiale: {initial_memory:.1f} MB")
    print(f"  📊 Mémoire finale: {final_memory:.1f} MB")
    print(f"  📊 Augmentation: {memory_increase:.1f} MB")
    print(f"  📊 Cache size: {cache_stats['cache_size']}")
    print(f"  📊 Mémoire par entrée cache: {memory_increase / cache_stats['cache_size']:.2f} MB")

def main():
    """Fonction principale de test."""
    print("🧪 Test Complet du Cache Intelligent")
    print("=" * 60)
    
    try:
        # Test 1: Performance du cache
        perf_results = test_cache_performance()
        
        # Test 2: Cache avec données différentes
        test_cache_with_different_data()
        
        # Test 3: Utilisation mémoire
        test_cache_memory_usage()
        
        # Résumé final
        print("\n" + "=" * 60)
        print("📋 RÉSUMÉ DES TESTS CACHE")
        print("=" * 60)
        
        print(f"✅ Accélération cache: {perf_results['cache_acceleration']:.1f}x")
        print(f"✅ Efficacité cache: {perf_results['cache_efficiency']:.1f}%")
        print(f"✅ Taux de hit: {perf_results['hit_rate']:.1%}")
        print(f"✅ Résultats cohérents: {'Oui' if perf_results['results_identical'] else 'Non'}")
        
        if (perf_results['cache_acceleration'] > 2.0 and 
            perf_results['cache_efficiency'] > 50.0 and
            perf_results['results_identical']):
            print("\n🎉 Cache intelligent opérationnel et performant !")
            return True
        else:
            print("\n⚠️ Performance du cache insuffisante.")
            return False
            
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)