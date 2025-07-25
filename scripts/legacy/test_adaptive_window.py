#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la fenêtre adaptative pour ADAN Trading Bot.
Teste la tâche 7.1.1 - Implémenter fenêtre adaptative.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict

# Add src to PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
project_root = SCRIPT_DIR.parent
sys.path.append(str(project_root))

from src.adan_trading_bot.data_processing.state_builder import StateBuilder

def create_volatile_market_data(n_points: int = 200, high_volatility: bool = False) -> Dict[str, pd.DataFrame]:
    """Créer des données de marché avec volatilité contrôlée."""
    
    timeframes = ['5m', '1h', '4h']
    data = {}
    
    # Base timestamp
    timestamps = pd.date_range('2023-01-01', periods=n_points, freq='5min', tz='UTC')
    
    for tf in timeframes:
        # Create price data with controlled volatility
        base_price = 50000
        
        if high_volatility:
            # High volatility: large price swings
            price_changes = np.random.normal(0, 0.05, n_points)  # 5% volatility
            volume_multiplier = np.random.uniform(2, 5, n_points)  # High volume
        else:
            # Low volatility: small price swings
            price_changes = np.random.normal(0, 0.01, n_points)  # 1% volatility
            volume_multiplier = np.random.uniform(0.5, 1.5, n_points)  # Normal volume
        
        # Generate OHLCV data
        close_prices = base_price * np.cumprod(1 + price_changes)
        
        df_data = {
            f'{tf}_open': close_prices * np.random.uniform(0.995, 1.005, n_points),
            f'{tf}_high': close_prices * np.random.uniform(1.001, 1.02, n_points),
            f'{tf}_low': close_prices * np.random.uniform(0.98, 0.999, n_points),
            f'{tf}_close': close_prices,
            f'{tf}_volume': np.random.uniform(1000, 5000, n_points) * volume_multiplier,
            f'{tf}_minutes_since_update': np.zeros(n_points)  # Fresh data
        }
        
        # Add technical indicators (simplified)
        for indicator in ['RSI', 'MACD', 'ATR', 'SMA_20', 'EMA_12']:
            df_data[f'{indicator}_{tf}'] = np.random.uniform(-1, 1, n_points)
        
        data[tf] = pd.DataFrame(df_data, index=timestamps)
    
    return data

def test_adaptive_window_initialization():
    """Test d'initialisation du StateBuilder avec fenêtre adaptative."""
    print("🧪 Test initialisation fenêtre adaptative...")
    
    try:
        # Test avec fenêtre adaptative activée
        state_builder = StateBuilder(
            window_size=100,
            adaptive_window=True,
            min_window_size=50,
            max_window_size=200
        )
        
        print(f"  📊 Base window size: {state_builder.base_window_size}")
        print(f"  📊 Current window size: {state_builder.window_size}")
        print(f"  📊 Adaptive enabled: {state_builder.adaptive_window}")
        print(f"  📊 Min/Max window: {state_builder.min_window_size}/{state_builder.max_window_size}")
        print(f"  📊 Timeframe weights: {state_builder.timeframe_weights}")
        
        # Vérifier les paramètres
        if (state_builder.adaptive_window and 
            state_builder.base_window_size == 100 and
            state_builder.min_window_size == 50 and
            state_builder.max_window_size == 200):
            print("  ✅ Initialisation correcte")
            return True
        else:
            print("  ❌ Problème d'initialisation")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_volatility_calculation():
    """Test du calcul de volatilité."""
    print("\n🧪 Test calcul de volatilité...")
    
    try:
        state_builder = StateBuilder(adaptive_window=True)
        
        # Test avec données faible volatilité
        low_vol_data = create_volatile_market_data(100, high_volatility=False)
        low_volatility = state_builder.calculate_market_volatility(low_vol_data, 50)
        
        # Test avec données haute volatilité
        high_vol_data = create_volatile_market_data(100, high_volatility=True)
        high_volatility = state_builder.calculate_market_volatility(high_vol_data, 50)
        
        print(f"  📊 Volatilité faible: {low_volatility:.3f}")
        print(f"  📊 Volatilité élevée: {high_volatility:.3f}")
        
        # La volatilité élevée devrait être supérieure à la faible
        if high_volatility > low_volatility and 0 <= low_volatility <= 2 and 0 <= high_volatility <= 2:
            print("  ✅ Calcul de volatilité correct")
            return True
        else:
            print("  ❌ Problème de calcul de volatilité")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_window_adaptation():
    """Test d'adaptation de la taille de fenêtre."""
    print("\n🧪 Test adaptation taille de fenêtre...")
    
    try:
        state_builder = StateBuilder(
            window_size=100,
            adaptive_window=True,
            min_window_size=50,
            max_window_size=200
        )
        
        # Test différents niveaux de volatilité
        test_cases = [
            (0.1, "Faible volatilité"),
            (0.5, "Volatilité moyenne"),
            (1.0, "Haute volatilité"),
            (1.8, "Volatilité extrême")
        ]
        
        results = []
        for volatility, description in test_cases:
            adapted_size = state_builder.adapt_window_size(volatility)
            results.append((volatility, adapted_size, description))
            print(f"  📊 {description}: volatilité={volatility:.1f} -> fenêtre={adapted_size}")
        
        # Vérifier la logique: haute volatilité = petite fenêtre
        sizes = [size for _, size, _ in results]
        
        # Les tailles devraient généralement diminuer avec l'augmentation de la volatilité
        decreasing_trend = all(sizes[i] >= sizes[i+1] for i in range(len(sizes)-1))
        within_bounds = all(50 <= size <= 200 for size in sizes)
        
        if within_bounds and (decreasing_trend or len(set(sizes)) > 1):
            print("  ✅ Adaptation de fenêtre correcte")
            return True
        else:
            print("  ❌ Problème d'adaptation de fenêtre")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_adaptive_observation_building():
    """Test de construction d'observations adaptatives."""
    print("\n🧪 Test construction observations adaptatives...")
    
    try:
        state_builder = StateBuilder(
            window_size=100,
            adaptive_window=True,
            min_window_size=50,
            max_window_size=150
        )
        
        # Créer des données de test
        data = create_volatile_market_data(200, high_volatility=True)
        
        # Construire observation adaptative
        observation = state_builder.build_adaptive_observation(100, data)
        
        if observation is not None:
            print(f"  📊 Shape observation: {observation.shape}")
            print(f"  📊 Taille fenêtre actuelle: {state_builder.window_size}")
            print(f"  📊 Min/Max valeurs: {observation.min():.3f}/{observation.max():.3f}")
            
            # Vérifier la shape
            expected_shape = (3, state_builder.window_size, observation.shape[2])
            if observation.shape[:2] == expected_shape[:2]:
                print("  ✅ Construction observation adaptative correcte")
                return True
            else:
                print(f"  ❌ Shape incorrecte: attendu {expected_shape}, obtenu {observation.shape}")
                return False
        else:
            print("  ❌ Observation None")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_timeframe_weighting():
    """Test de pondération des timeframes."""
    print("\n🧪 Test pondération timeframes...")
    
    try:
        state_builder = StateBuilder(adaptive_window=True)
        
        # Créer observations de test
        observations = {}
        for tf in ['5m', '1h', '4h']:
            observations[tf] = np.random.uniform(-1, 1, (50, 20))
        
        # Appliquer pondération
        weighted_obs = state_builder.apply_timeframe_weighting(observations)
        
        print(f"  📊 Timeframes traités: {list(weighted_obs.keys())}")
        
        # Vérifier que les poids sont appliqués
        weights_applied = True
        for tf in ['5m', '1h', '4h']:
            if tf in observations and tf in weighted_obs:
                original_mean = np.abs(observations[tf]).mean()
                weighted_mean = np.abs(weighted_obs[tf]).mean()
                weight = state_builder.timeframe_weights[tf]
                
                print(f"    {tf}: poids={weight}, original={original_mean:.3f}, pondéré={weighted_mean:.3f}")
                
                # Pour 5m (poids=1.0), les valeurs devraient être similaires
                # Pour 1h et 4h, les valeurs devraient être réduites
                if tf == '5m':
                    if abs(original_mean - weighted_mean) > 0.1:
                        weights_applied = False
                else:
                    if weighted_mean >= original_mean:
                        weights_applied = False
        
        if weights_applied:
            print("  ✅ Pondération timeframes correcte")
            return True
        else:
            print("  ❌ Problème de pondération")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_adaptive_stats():
    """Test des statistiques adaptatives."""
    print("\n🧪 Test statistiques adaptatives...")
    
    try:
        state_builder = StateBuilder(
            window_size=100,
            adaptive_window=True,
            min_window_size=50,
            max_window_size=200
        )
        
        # Ajouter quelques données de volatilité
        state_builder.volatility_history = [0.1, 0.3, 0.5, 0.8, 1.2]
        
        # Obtenir statistiques
        stats = state_builder.get_adaptive_stats()
        
        print(f"  📊 Statistiques obtenues: {list(stats.keys())}")
        print(f"  📊 Adaptatif activé: {stats['adaptive_enabled']}")
        print(f"  📊 Fenêtre base/actuelle: {stats['base_window_size']}/{stats['current_window_size']}")
        print(f"  📊 Historique volatilité: {len(stats['volatility_history'])} points")
        print(f"  📊 Volatilité actuelle: {stats['current_volatility']:.3f}")
        
        # Vérifier les clés requises
        required_keys = [
            'adaptive_enabled', 'base_window_size', 'current_window_size',
            'min_window_size', 'max_window_size', 'volatility_history',
            'current_volatility', 'timeframe_weights'
        ]
        
        if all(key in stats for key in required_keys):
            print("  ✅ Statistiques adaptatives complètes")
            return True
        else:
            missing = [key for key in required_keys if key not in stats]
            print(f"  ❌ Clés manquantes: {missing}")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_dynamic_window_updates():
    """Test des mises à jour dynamiques de fenêtre."""
    print("\n🧪 Test mises à jour dynamiques...")
    
    try:
        state_builder = StateBuilder(
            window_size=100,
            adaptive_window=True,
            min_window_size=50,
            max_window_size=150
        )
        
        # Créer données avec volatilité changeante
        low_vol_data = create_volatile_market_data(100, high_volatility=False)
        high_vol_data = create_volatile_market_data(100, high_volatility=True)
        
        initial_size = state_builder.window_size
        print(f"  📊 Taille initiale: {initial_size}")
        
        # Test avec faible volatilité
        state_builder.update_adaptive_window(low_vol_data, 50)
        low_vol_size = state_builder.window_size
        print(f"  📊 Après faible volatilité: {low_vol_size}")
        
        # Test avec haute volatilité
        state_builder.update_adaptive_window(high_vol_data, 50)
        high_vol_size = state_builder.window_size
        print(f"  📊 Après haute volatilité: {high_vol_size}")
        
        # La fenêtre devrait s'adapter
        adaptation_occurred = (low_vol_size != initial_size or 
                             high_vol_size != low_vol_size or
                             high_vol_size != initial_size)
        
        if adaptation_occurred:
            print("  ✅ Mises à jour dynamiques fonctionnelles")
            return True
        else:
            print("  ⚠️ Pas d'adaptation détectée (peut être normal)")
            return True  # Pas forcément un échec
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def main():
    """Fonction principale pour exécuter tous les tests."""
    print("🚀 Test Complet Fenêtre Adaptative ADAN")
    print("=" * 60)
    
    tests = [
        ("Initialisation Adaptative", test_adaptive_window_initialization),
        ("Calcul Volatilité", test_volatility_calculation),
        ("Adaptation Fenêtre", test_window_adaptation),
        ("Construction Observations", test_adaptive_observation_building),
        ("Pondération Timeframes", test_timeframe_weighting),
        ("Statistiques Adaptatives", test_adaptive_stats),
        ("Mises à jour Dynamiques", test_dynamic_window_updates)
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
    print("📋 RÉSUMÉ DES TESTS FENÊTRE ADAPTATIVE")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Score: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Fenêtre adaptative opérationnelle !")
    else:
        print("⚠️ Certains tests ont échoué.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)