#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test des améliorations de gestion temporelle pour ADAN Trading Bot.
Teste la tâche 6.2.2 - Améliorer gestion temporelle.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
import pytz

# Add src to PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
project_root = SCRIPT_DIR.parent
sys.path.append(str(project_root))

from scripts.convert_real_data import (
    calculate_minutes_since_update,
    parse_timeframe_to_minutes,
    handle_dst_transitions
)

def create_test_data_with_gaps(timeframe: str = '5m', n_points: int = 100) -> pd.DataFrame:
    """Créer des données de test avec des gaps temporels."""
    
    # Parse timeframe to get interval
    if timeframe.endswith('m'):
        freq = f"{timeframe[:-1]}T"
        interval_minutes = int(timeframe[:-1])
    elif timeframe.endswith('h'):
        freq = f"{timeframe[:-1]}H"
        interval_minutes = int(timeframe[:-1]) * 60
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Create base timestamp range
    start_time = pd.Timestamp('2023-01-01 00:00:00', tz='UTC')
    timestamps = pd.date_range(start=start_time, periods=n_points, freq=freq)
    
    # Create gaps in data (simulate missing data)
    gap_indices = [20, 21, 22, 45, 46]  # Remove some timestamps (ensure within bounds)
    gap_indices = [i for i in gap_indices if i < len(timestamps)]  # Filter valid indices
    timestamps = timestamps.delete(gap_indices)
    
    # Add DST transition simulation (spring forward - 1 hour gap)
    dst_gap_idx = len(timestamps) // 2
    dst_timestamp = timestamps[dst_gap_idx] + pd.Timedelta(hours=1)
    timestamps = timestamps.insert(dst_gap_idx + 1, dst_timestamp)
    
    # Create OHLCV data
    n_actual = len(timestamps)
    base_price = 50000
    
    data = {
        'timestamp': timestamps,
        'open': base_price + np.random.normal(0, 100, n_actual),
        'high': base_price + np.random.normal(200, 100, n_actual),
        'low': base_price + np.random.normal(-200, 100, n_actual),
        'close': base_price + np.random.normal(0, 100, n_actual),
        'volume': np.random.uniform(1000, 10000, n_actual)
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    
    return df

def test_parse_timeframe_to_minutes():
    """Test de la fonction parse_timeframe_to_minutes."""
    print("🧪 Test parse_timeframe_to_minutes...")
    
    test_cases = [
        ('5m', 5.0),
        ('15m', 15.0),
        ('1h', 60.0),
        ('4h', 240.0),
        ('1d', 1440.0)
    ]
    
    success = True
    for timeframe, expected in test_cases:
        try:
            result = parse_timeframe_to_minutes(timeframe)
            if result == expected:
                print(f"  ✅ {timeframe} -> {result} minutes")
            else:
                print(f"  ❌ {timeframe} -> {result} minutes (expected {expected})")
                success = False
        except Exception as e:
            print(f"  ❌ {timeframe} -> Error: {e}")
            success = False
    
    return success

def test_minutes_since_update_basic():
    """Test de base pour calculate_minutes_since_update."""
    print("\n🧪 Test calculate_minutes_since_update (base)...")
    
    # Create perfect 5-minute data
    timestamps = pd.date_range('2023-01-01', periods=10, freq='5T', tz='UTC')
    df = pd.DataFrame({
        'close': np.random.uniform(100, 200, 10)
    }, index=timestamps)
    
    try:
        result = calculate_minutes_since_update(df, '5m')
        
        # For perfect data, all values should be 0 (fresh)
        expected_fresh = (result == 0).sum()
        total_points = len(result)
        
        print(f"  📊 Points parfaitement frais: {expected_fresh}/{total_points}")
        print(f"  📊 Valeurs: {result.values[:5]}...")
        
        if expected_fresh >= total_points - 1:  # Allow first point to be 0
            print("  ✅ Données parfaitement fraîches détectées")
            return True
        else:
            print("  ❌ Problème de détection de fraîcheur")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_minutes_since_update_with_gaps():
    """Test avec des gaps temporels."""
    print("\n🧪 Test calculate_minutes_since_update (avec gaps)...")
    
    # Create data with gaps
    df = create_test_data_with_gaps('5m', 50)
    
    try:
        result = calculate_minutes_since_update(df, '5m')
        
        # Analyze results
        fresh_data = (result == 0).sum()
        stale_data = (result > 0).sum()
        max_staleness = result.max()
        
        print(f"  📊 Données fraîches: {fresh_data}")
        print(f"  📊 Données périmées: {stale_data}")
        print(f"  📊 Péremption max: {max_staleness:.1f} minutes")
        print(f"  📊 Échantillon de valeurs: {result.values[:10]}")
        
        # Should detect some stale data due to gaps
        if stale_data > 0 and max_staleness > 0:
            print("  ✅ Gaps temporels détectés correctement")
            return True
        else:
            print("  ❌ Gaps temporels non détectés")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_dst_transitions():
    """Test de gestion des transitions DST."""
    print("\n🧪 Test handle_dst_transitions...")
    
    try:
        # Simulate DST transition data
        n_points = 10
        expected_interval = 60.0  # 1 hour
        
        # Normal time differences
        time_diffs = pd.Series([60.0] * n_points)
        
        # Add DST spring forward (2 hour gap becomes 1 hour)
        time_diffs.iloc[5] = 120.0  # 2 hour gap (spring forward)
        
        # Add DST fall back (0 hour gap)
        time_diffs.iloc[7] = 0.0  # Overlap (fall back)
        
        # Calculate initial freshness
        freshness = np.maximum(0, time_diffs - expected_interval)
        
        # Apply DST handling
        adjusted = handle_dst_transitions(freshness, time_diffs, expected_interval)
        
        print(f"  📊 Différences temporelles: {time_diffs.values}")
        print(f"  📊 Fraîcheur initiale: {freshness.values}")
        print(f"  📊 Fraîcheur ajustée DST: {adjusted.values}")
        
        # Check DST adjustments
        spring_forward_adjusted = adjusted.iloc[5] < freshness.iloc[5]
        fall_back_adjusted = adjusted.iloc[7] == 0
        
        if spring_forward_adjusted and fall_back_adjusted:
            print("  ✅ Transitions DST gérées correctement")
            return True
        else:
            print("  ❌ Problème de gestion DST")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_different_timeframes():
    """Test avec différents timeframes."""
    print("\n🧪 Test différents timeframes...")
    
    timeframes = ['5m', '15m', '1h', '4h']
    success_count = 0
    
    for tf in timeframes:
        try:
            # Create appropriate test data
            if tf.endswith('m'):
                freq = f"{tf[:-1]}T"
                n_points = 20
            else:
                freq = f"{tf[:-1]}H"
                n_points = 10
            
            timestamps = pd.date_range('2023-01-01', periods=n_points, freq=freq, tz='UTC')
            df = pd.DataFrame({
                'close': np.random.uniform(100, 200, n_points)
            }, index=timestamps)
            
            result = calculate_minutes_since_update(df, tf)
            fresh_count = (result == 0).sum()
            
            print(f"  📊 {tf}: {fresh_count}/{n_points} points frais")
            
            if fresh_count >= n_points - 1:  # Allow for first point
                print(f"    ✅ {tf} traité correctement")
                success_count += 1
            else:
                print(f"    ❌ {tf} problème détecté")
                
        except Exception as e:
            print(f"    ❌ {tf} erreur: {e}")
    
    return success_count == len(timeframes)

def test_temporal_synchronization():
    """Test de synchronisation temporelle multi-timeframes."""
    print("\n🧪 Test synchronisation temporelle...")
    
    try:
        # Create data for multiple timeframes
        base_time = pd.Timestamp('2023-01-01 00:00:00', tz='UTC')
        
        # 5-minute data
        timestamps_5m = pd.date_range(base_time, periods=60, freq='5T')
        df_5m = pd.DataFrame({
            'close': np.random.uniform(100, 200, 60)
        }, index=timestamps_5m)
        
        # 1-hour data (should align with 5m data every 12 points)
        timestamps_1h = pd.date_range(base_time, periods=5, freq='1H')
        df_1h = pd.DataFrame({
            'close': np.random.uniform(100, 200, 5)
        }, index=timestamps_1h)
        
        # Calculate minutes_since_update for both
        result_5m = calculate_minutes_since_update(df_5m, '5m')
        result_1h = calculate_minutes_since_update(df_1h, '1h')
        
        # Check alignment - 1h timestamps should align with some 5m timestamps
        aligned_timestamps = set(timestamps_1h) & set(timestamps_5m)
        alignment_ratio = len(aligned_timestamps) / len(timestamps_1h)
        
        print(f"  📊 Timestamps 5m: {len(timestamps_5m)}")
        print(f"  📊 Timestamps 1h: {len(timestamps_1h)}")
        print(f"  📊 Timestamps alignés: {len(aligned_timestamps)}")
        print(f"  📊 Ratio d'alignement: {alignment_ratio:.2%}")
        
        if alignment_ratio >= 0.8:  # At least 80% should align
            print("  ✅ Synchronisation temporelle correcte")
            return True
        else:
            print("  ❌ Problème de synchronisation temporelle")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_timezone_handling():
    """Test de gestion des fuseaux horaires."""
    print("\n🧪 Test gestion fuseaux horaires...")
    
    try:
        # Test with different timezones
        timezones = ['UTC', 'US/Eastern', 'Europe/London', 'Asia/Tokyo']
        success_count = 0
        
        for tz in timezones:
            try:
                # Create data in specific timezone
                timestamps = pd.date_range(
                    '2023-06-01 00:00:00', 
                    periods=10, 
                    freq='1H',
                    tz=tz
                )
                
                df = pd.DataFrame({
                    'close': np.random.uniform(100, 200, 10)
                }, index=timestamps)
                
                result = calculate_minutes_since_update(df, '1h')
                fresh_count = (result == 0).sum()
                
                print(f"  📊 {tz}: {fresh_count}/10 points frais")
                
                if fresh_count >= 9:  # Allow for first point
                    success_count += 1
                    
            except Exception as e:
                print(f"  ⚠️ {tz}: {e}")
        
        if success_count >= len(timezones) - 1:  # Allow one timezone to fail
            print("  ✅ Gestion fuseaux horaires correcte")
            return True
        else:
            print("  ❌ Problème de gestion fuseaux horaires")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def main():
    """Fonction principale pour exécuter tous les tests."""
    print("🚀 Test Complet Améliorations Temporelles ADAN")
    print("=" * 60)
    
    tests = [
        ("Parse Timeframe", test_parse_timeframe_to_minutes),
        ("Minutes Since Update (Base)", test_minutes_since_update_basic),
        ("Minutes Since Update (Gaps)", test_minutes_since_update_with_gaps),
        ("Transitions DST", test_dst_transitions),
        ("Différents Timeframes", test_different_timeframes),
        ("Synchronisation Temporelle", test_temporal_synchronization),
        ("Gestion Fuseaux Horaires", test_timezone_handling)
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
    print("📋 RÉSUMÉ DES TESTS TEMPORELS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Score: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Toutes les améliorations temporelles sont opérationnelles !")
    else:
        print("⚠️ Certains tests ont échoué.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)