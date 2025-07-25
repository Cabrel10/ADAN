#!/usr/bin/env python3
"""
Script de test pour valider les optimisations du DataLoader pour 10+ paires de trading.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.data_processing.data_loader import ComprehensiveDataLoader

def create_test_parquet_data(temp_dir: Path, assets: list, timeframes: list, num_rows: int = 1000):
    """Créer des données de test au format parquet."""
    print(f"📊 Création de données de test pour {len(assets)} assets...")
    
    for asset in assets:
        for tf in timeframes:
            # Créer le répertoire pour le timeframe
            tf_dir = temp_dir / tf
            tf_dir.mkdir(parents=True, exist_ok=True)
            
            # Générer des données OHLCV réalistes
            np.random.seed(hash(asset + tf) % 2**32)  # Seed basée sur asset et timeframe
            
            base_price = np.random.uniform(1000, 50000)  # Prix de base aléatoire
            returns = np.random.normal(0, 0.02, num_rows)
            prices = [base_price]
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            prices = np.array(prices[1:])
            
            # Créer DataFrame OHLCV
            data = {
                'timestamp': pd.date_range(start='2023-01-01', periods=num_rows, freq='5T'),
                'open': prices * (1 + np.random.normal(0, 0.001, num_rows)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, num_rows))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, num_rows))),
                'close': prices,
                'volume': np.random.lognormal(10, 1, num_rows)
            }
            
            df = pd.DataFrame(data)
            df['__index_level_0__'] = range(len(df))  # Index pour filtrage parquet
            
            # Sauvegarder en parquet
            file_path = tf_dir / f"{asset}.parquet"
            df.to_parquet(file_path, engine='fastparquet', index=False)
    
    print(f"✅ Données créées dans {temp_dir}")

def test_basic_functionality():
    """Test de la fonctionnalité de base du DataLoader optimisé."""
    print("🧪 Test Fonctionnalité de Base")
    print("=" * 50)
    
    # Créer répertoire temporaire
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Configuration de test
        assets = ['BTC', 'ETH', 'SOL']
        timeframes = ['5m', '1h', '4h']
        
        # Créer données de test
        create_test_parquet_data(temp_dir, assets, timeframes, 500)
        
        # Configuration du DataLoader
        config = {
            'feature_engineering': {
                'timeframes': timeframes
            },
            'data_sources': [
                {'assets': assets}
            ],
            'chunk_size': 100,
            'enable_cache': True,
            'cache_size': 10,
            'enable_parallel_loading': True,
            'max_workers': 2
        }
        
        # Créer DataLoader
        loader = ComprehensiveDataLoader(config, str(temp_dir))
        
        # Charger les chemins
        loader.load_asset_paths()
        
        print(f"📊 Assets détectés: {loader.assets}")
        print(f"📊 Timeframes: {loader.timeframes}")
        print(f"📊 Total rows par asset: {loader.asset_total_rows}")
        
        # Tester chargement de chunks
        chunks_loaded = 0
        total_rows = 0
        
        while True:
            chunk = loader.get_next_chunk()
            if chunk is None:
                break
            
            chunks_loaded += 1
            total_rows += len(chunk)
            
            if chunks_loaded <= 3:  # Afficher détails des premiers chunks
                print(f"  Chunk {chunks_loaded}: {len(chunk)} rows, "
                      f"Asset: {loader.current_asset}, "
                      f"Colonnes: {len(chunk.columns)}")
        
        print(f"✅ Total chunks chargés: {chunks_loaded}")
        print(f"✅ Total rows traitées: {total_rows}")
        
        return chunks_loaded > 0 and total_rows > 0
        
    finally:
        # Nettoyer
        shutil.rmtree(temp_dir)

def test_cache_performance():
    """Test des performances du cache intelligent."""
    print("\n🚀 Test Performance Cache")
    print("=" * 50)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Configuration avec plus d'assets pour tester le cache
        assets = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP']
        timeframes = ['5m', '1h', '4h']
        
        create_test_parquet_data(temp_dir, assets, timeframes, 300)
        
        # Test avec cache activé
        config_with_cache = {
            'feature_engineering': {'timeframes': timeframes},
            'data_sources': [{'assets': assets}],
            'chunk_size': 50,
            'enable_cache': True,
            'cache_size': 20,
            'enable_parallel_loading': True,
            'max_workers': 3
        }
        
        loader_cached = ComprehensiveDataLoader(config_with_cache, str(temp_dir))
        loader_cached.load_asset_paths()
        
        # Premier passage (cache miss)
        start_time = time.time()
        chunks_first_pass = []
        
        for _ in range(10):  # Charger 10 chunks
            chunk = loader_cached.get_next_chunk()
            if chunk is None:
                break
            chunks_first_pass.append(len(chunk))
        
        first_pass_time = time.time() - start_time
        
        # Reset et deuxième passage (cache hit)
        loader_cached.reset()
        start_time = time.time()
        chunks_second_pass = []
        
        for _ in range(10):  # Recharger les mêmes 10 chunks
            chunk = loader_cached.get_next_chunk()
            if chunk is None:
                break
            chunks_second_pass.append(len(chunk))
        
        second_pass_time = time.time() - start_time
        
        # Statistiques du cache
        cache_stats = loader_cached.get_cache_statistics()
        
        print(f"📊 Premier passage (cache miss): {first_pass_time:.3f}s")
        print(f"📊 Deuxième passage (cache hit): {second_pass_time:.3f}s")
        print(f"🚀 Accélération: {first_pass_time / second_pass_time:.1f}x")
        print(f"📈 Cache hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"📊 Cache size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        
        return (first_pass_time / second_pass_time) > 2.0  # Au moins 2x plus rapide
        
    finally:
        shutil.rmtree(temp_dir)

def test_parallel_loading():
    """Test du chargement parallèle."""
    print("\n⚡ Test Chargement Parallèle")
    print("=" * 50)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        assets = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOT', 'LINK', 'UNI']
        timeframes = ['5m', '1h', '4h']
        
        create_test_parquet_data(temp_dir, assets, timeframes, 200)
        
        # Test avec chargement parallèle
        config_parallel = {
            'feature_engineering': {'timeframes': timeframes},
            'data_sources': [{'assets': assets}],
            'chunk_size': 50,
            'enable_cache': False,  # Désactiver cache pour test pur
            'enable_parallel_loading': True,
            'max_workers': 4
        }
        
        loader_parallel = ComprehensiveDataLoader(config_parallel, str(temp_dir))
        loader_parallel.load_asset_paths()
        
        start_time = time.time()
        parallel_chunks = 0
        
        for _ in range(15):  # Charger 15 chunks
            chunk = loader_parallel.get_next_chunk()
            if chunk is None:
                break
            parallel_chunks += 1
        
        parallel_time = time.time() - start_time
        
        # Test avec chargement séquentiel
        config_sequential = config_parallel.copy()
        config_sequential['enable_parallel_loading'] = False
        
        loader_sequential = ComprehensiveDataLoader(config_sequential, str(temp_dir))
        loader_sequential.load_asset_paths()
        
        start_time = time.time()
        sequential_chunks = 0
        
        for _ in range(15):  # Charger 15 chunks
            chunk = loader_sequential.get_next_chunk()
            if chunk is None:
                break
            sequential_chunks += 1
        
        sequential_time = time.time() - start_time
        
        print(f"📊 Chargement parallèle: {parallel_time:.3f}s ({parallel_chunks} chunks)")
        print(f"📊 Chargement séquentiel: {sequential_time:.3f}s ({sequential_chunks} chunks)")
        print(f"🚀 Accélération parallèle: {sequential_time / parallel_time:.1f}x")
        
        return sequential_time > parallel_time  # Parallèle doit être plus rapide
        
    finally:
        shutil.rmtree(temp_dir)

def test_memory_management():
    """Test de la gestion mémoire."""
    print("\n💾 Test Gestion Mémoire")
    print("=" * 50)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Créer beaucoup d'assets pour tester la gestion mémoire
        assets = [f'ASSET_{i:02d}' for i in range(12)]  # 12 assets
        timeframes = ['5m', '1h', '4h']
        
        create_test_parquet_data(temp_dir, assets, timeframes, 400)
        
        config = {
            'feature_engineering': {'timeframes': timeframes},
            'data_sources': [{'assets': assets}],
            'chunk_size': 100,
            'enable_cache': True,
            'cache_size': 30,
            'memory_threshold_mb': 500,  # Seuil bas pour test
            'enable_parallel_loading': True,
            'max_workers': 4
        }
        
        loader = ComprehensiveDataLoader(config, str(temp_dir))
        loader.load_asset_paths()
        
        print(f"📊 Assets créés: {len(assets)}")
        print(f"📊 Mémoire initiale: {loader.get_memory_usage_mb():.1f}MB")
        
        # Charger plusieurs chunks et surveiller la mémoire
        chunks_loaded = 0
        memory_readings = []
        
        for _ in range(25):  # Charger beaucoup de chunks
            chunk = loader.get_next_chunk()
            if chunk is None:
                break
            
            chunks_loaded += 1
            memory_mb = loader.get_memory_usage_mb()
            memory_readings.append(memory_mb)
            
            # Optimiser le cache périodiquement
            if chunks_loaded % 5 == 0:
                loader.optimize_cache_size()
        
        # Statistiques finales
        perf_stats = loader.get_performance_statistics()
        
        print(f"📊 Chunks chargés: {chunks_loaded}")
        print(f"📊 Mémoire finale: {memory_readings[-1]:.1f}MB")
        print(f"📊 Mémoire max: {max(memory_readings):.1f}MB")
        print(f"📊 Cache final: {perf_stats['cache_stats']['cache_size']}")
        print(f"📊 Hit rate: {perf_stats['cache_stats']['hit_rate']:.1%}")
        
        return chunks_loaded > 20 and max(memory_readings) < 1000  # Pas trop de mémoire
        
    finally:
        shutil.rmtree(temp_dir)

def test_scalability_10_plus_assets():
    """Test de scalabilité avec 10+ assets."""
    print("\n📈 Test Scalabilité 10+ Assets")
    print("=" * 50)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Créer 15 assets pour tester la scalabilité
        assets = [
            'BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOT', 'LINK', 'UNI', 
            'AVAX', 'MATIC', 'ATOM', 'NEAR', 'FTM', 'ALGO', 'ICP'
        ]
        timeframes = ['5m', '1h', '4h']
        
        create_test_parquet_data(temp_dir, assets, timeframes, 300)
        
        config = {
            'feature_engineering': {'timeframes': timeframes},
            'data_sources': [{'assets': assets}],
            'chunk_size': 75,
            'enable_cache': True,
            'cache_size': 50,
            'memory_threshold_mb': 2000,
            'enable_parallel_loading': True,
            'max_workers': 6,
            'preload_next_chunks': 3
        }
        
        loader = ComprehensiveDataLoader(config, str(temp_dir))
        
        # Mesurer le temps d'initialisation
        start_time = time.time()
        loader.load_asset_paths()
        init_time = time.time() - start_time
        
        print(f"📊 Assets: {len(assets)}")
        print(f"📊 Temps d'initialisation: {init_time:.3f}s")
        print(f"📊 Rows totales: {sum(loader.asset_total_rows.values())}")
        
        # Test de chargement avec préchargement
        start_time = time.time()
        chunks_loaded = 0
        total_rows = 0
        
        # Précharger quelques chunks
        loader.preload_chunks(5)
        
        # Charger des chunks de tous les assets
        for _ in range(30):  # Charger 30 chunks
            chunk = loader.get_next_chunk()
            if chunk is None:
                break
            
            chunks_loaded += 1
            total_rows += len(chunk)
            
            # Afficher progrès pour les premiers assets
            if chunks_loaded <= 5:
                print(f"  Chunk {chunks_loaded}: {len(chunk)} rows, "
                      f"Asset: {loader.current_asset}")
        
        total_time = time.time() - start_time
        
        # Statistiques finales
        perf_stats = loader.get_performance_statistics()
        
        print(f"📊 Chunks chargés: {chunks_loaded}")
        print(f"📊 Rows traitées: {total_rows}")
        print(f"📊 Temps total: {total_time:.3f}s")
        print(f"📊 Vitesse: {total_rows / total_time:.0f} rows/sec")
        print(f"📊 Cache hit rate: {perf_stats['cache_stats']['hit_rate']:.1%}")
        print(f"📊 Temps moyen/chunk: {perf_stats.get('avg_load_time', 0):.4f}s")
        
        return (chunks_loaded >= 25 and 
                total_rows > 1000 and 
                perf_stats['cache_stats']['hit_rate'] > 0.1)
        
    finally:
        shutil.rmtree(temp_dir)

def main():
    """Fonction principale de test."""
    print("🚀 Test Complet DataLoader Optimisé pour 10+ Paires")
    print("=" * 60)
    
    tests = [
        ("Fonctionnalité de base", test_basic_functionality),
        ("Performance cache", test_cache_performance),
        ("Chargement parallèle", test_parallel_loading),
        ("Gestion mémoire", test_memory_management),
        ("Scalabilité 10+ assets", test_scalability_10_plus_assets)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
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
    print("📋 RÉSUMÉ DES TESTS DATALOADER")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Score: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 DataLoader optimisé opérationnel pour 10+ paires !")
        return True
    else:
        print("⚠️ Certains tests ont échoué.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)