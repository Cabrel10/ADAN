#!/usr/bin/env python3
"""
Test du système de cache intelligent pour ADAN Trading Bot.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add src to path
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

from adan_trading_bot.utils.intelligent_cache import (
    IntelligentCache, cached, get_global_cache_stats, clear_global_cache,
    observation_cache, indicator_cache
)


class CachePerformanceTester:
    """Testeur de performance pour le système de cache"""
    
    def __init__(self):
        self.test_results = {}
        self.test_cache = IntelligentCache(memory_cache_size=100)
    
    def _expensive_computation(self, data: np.ndarray, window: int) -> np.ndarray:
        """Simulation d'un calcul coûteux"""
        time.sleep(0.1)  # Simulate expensive computation
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    @cached(persist_to_disk=True)
    def cached_expensive_computation(self, data: np.ndarray, window: int) -> np.ndarray:
        """Version cachée du calcul coûteux"""
        time.sleep(0.1)  # Simulate expensive computation
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    def test_basic_caching(self) -> Dict[str, Any]:
        """Test basique du système de cache"""
        print("🧪 Test basique du cache...")
        
        # Generate test data
        data = np.random.randn(1000)
        window = 20
        
        # First call (cache miss)
        start_time = time.time()
        result1 = self.cached_expensive_computation(data, window)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = self.cached_expensive_computation(data, window)
        second_call_time = time.time() - start_time
        
        # Verify results are identical
        results_match = np.allclose(result1, result2)
        
        # Calculate speedup
        speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
        
        return {
            'test_name': 'basic_caching',
            'first_call_time': first_call_time,
            'second_call_time': second_call_time,
            'speedup': speedup,
            'results_match': results_match,
            'cache_stats': get_global_cache_stats()
        }
    
    def test_memory_vs_disk_cache(self) -> Dict[str, Any]:
        """Test performance mémoire vs disque"""
        print("💾 Test cache mémoire vs disque...")
        
        # Clear cache first
        clear_global_cache()
        
        data = np.random.randn(500)
        
        # Test multiple different computations to fill cache
        computation_times = []
        
        for i in range(10):
            window = 10 + i * 2
            
            # First call (computation + cache storage)
            start_time = time.time()
            result = self.cached_expensive_computation(data, window)
            computation_times.append(time.time() - start_time)
        
        # Now test retrieval times
        retrieval_times = []
        
        for i in range(10):
            window = 10 + i * 2
            
            # Second call (cache retrieval)
            start_time = time.time()
            result = self.cached_expensive_computation(data, window)
            retrieval_times.append(time.time() - start_time)
        
        avg_computation_time = np.mean(computation_times)
        avg_retrieval_time = np.mean(retrieval_times)
        avg_speedup = avg_computation_time / avg_retrieval_time if avg_retrieval_time > 0 else float('inf')
        
        return {
            'test_name': 'memory_vs_disk',
            'avg_computation_time': avg_computation_time,
            'avg_retrieval_time': avg_retrieval_time,
            'avg_speedup': avg_speedup,
            'cache_stats': get_global_cache_stats()
        }
    
    def test_cache_with_different_data_types(self) -> Dict[str, Any]:
        """Test cache avec différents types de données"""
        print("🔢 Test cache avec différents types...")
        
        @cached(persist_to_disk=False)
        def process_numpy_array(arr: np.ndarray) -> float:
            time.sleep(0.05)
            return np.mean(arr)
        
        @cached(persist_to_disk=False)
        def process_dataframe(df: pd.DataFrame) -> float:
            time.sleep(0.05)
            return df.mean().mean()
        
        @cached(persist_to_disk=False)
        def process_mixed_args(arr: np.ndarray, scalar: float, text: str) -> str:
            time.sleep(0.05)
            return f"{text}_{np.sum(arr)}_{scalar}"
        
        # Test numpy array
        arr = np.random.randn(100)
        start_time = time.time()
        result1 = process_numpy_array(arr)
        first_numpy_time = time.time() - start_time
        
        start_time = time.time()
        result2 = process_numpy_array(arr)
        second_numpy_time = time.time() - start_time
        
        numpy_speedup = first_numpy_time / second_numpy_time if second_numpy_time > 0 else float('inf')
        
        # Test DataFrame
        df = pd.DataFrame(np.random.randn(50, 5))
        start_time = time.time()
        result3 = process_dataframe(df)
        first_df_time = time.time() - start_time
        
        start_time = time.time()
        result4 = process_dataframe(df)
        second_df_time = time.time() - start_time
        
        df_speedup = first_df_time / second_df_time if second_df_time > 0 else float('inf')
        
        # Test mixed arguments
        start_time = time.time()
        result5 = process_mixed_args(arr, 3.14, "test")
        first_mixed_time = time.time() - start_time
        
        start_time = time.time()
        result6 = process_mixed_args(arr, 3.14, "test")
        second_mixed_time = time.time() - start_time
        
        mixed_speedup = first_mixed_time / second_mixed_time if second_mixed_time > 0 else float('inf')
        
        return {
            'test_name': 'different_data_types',
            'numpy_speedup': numpy_speedup,
            'dataframe_speedup': df_speedup,
            'mixed_args_speedup': mixed_speedup,
            'results_match': {
                'numpy': result1 == result2,
                'dataframe': result3 == result4,
                'mixed': result5 == result6
            }
        }
    
    def test_cache_invalidation(self) -> Dict[str, Any]:
        """Test invalidation du cache"""
        print("🗑️ Test invalidation du cache...")
        
        # Create a simple cache instance for this test
        test_cache = IntelligentCache(memory_cache_size=10)
        
        @cached(persist_to_disk=False, cache_instance=test_cache)
        def simple_computation(x: float) -> float:
            time.sleep(0.01)
            return x * 2
        
        # Fill cache with some values
        for i in range(5):
            result = simple_computation(float(i))
        
        stats_before = test_cache.get_comprehensive_stats()
        
        # Clear cache
        test_cache.clear()
        
        stats_after = test_cache.get_comprehensive_stats()
        
        # Test that cache was cleared
        start_time = time.time()
        result = simple_computation(1.0)  # Should recompute
        recompute_time = time.time() - start_time
        
        return {
            'test_name': 'cache_invalidation',
            'cache_size_before': stats_before['memory_cache']['size'],
            'cache_size_after': stats_after['memory_cache']['size'],
            'recompute_time': recompute_time,
            'invalidation_successful': stats_after['memory_cache']['size'] == 0
        }
    
    def test_specialized_caches(self) -> Dict[str, Any]:
        """Test des caches spécialisés"""
        print("🎯 Test caches spécialisés...")
        
        # Test indicator cache
        data = np.random.randn(1000)
        indicator_name = "test_sma"
        window = 20
        
        # Simulate indicator calculation
        start_time = time.time()
        # First time - should compute
        cached_result = indicator_cache.get_cached_indicator(indicator_name, data, window=window)
        if cached_result is None:
            # Compute indicator
            result = np.convolve(data, np.ones(window)/window, mode='same')
            indicator_cache.put_cached_indicator(indicator_name, data, result, window=window)
            first_time = time.time() - start_time
        else:
            first_time = time.time() - start_time
        
        # Second time - should be cached
        start_time = time.time()
        cached_result = indicator_cache.get_cached_indicator(indicator_name, data, window=window)
        second_time = time.time() - start_time
        
        indicator_speedup = first_time / second_time if second_time > 0 else float('inf')
        
        return {
            'test_name': 'specialized_caches',
            'indicator_cache_hit': cached_result is not None,
            'indicator_speedup': indicator_speedup,
            'first_time': first_time,
            'second_time': second_time
        }
    
    def test_cache_memory_usage(self) -> Dict[str, Any]:
        """Test usage mémoire du cache"""
        print("📊 Test usage mémoire...")
        
        # Create cache with known size limit
        memory_cache = IntelligentCache(memory_cache_size=50)
        
        @cached(persist_to_disk=False, cache_instance=memory_cache)
        def generate_large_data(size: int) -> np.ndarray:
            return np.random.randn(size)
        
        # Fill cache with increasingly large arrays
        for i in range(10):
            size = 100 * (i + 1)
            result = generate_large_data(size)
        
        stats = memory_cache.get_comprehensive_stats()
        
        # Test LRU eviction by adding more items
        for i in range(60):  # More than cache size
            result = generate_large_data(i + 1000)
        
        final_stats = memory_cache.get_comprehensive_stats()
        
        return {
            'test_name': 'memory_usage',
            'initial_cache_size': stats['memory_cache']['size'],
            'final_cache_size': final_stats['memory_cache']['size'],
            'memory_usage_mb': final_stats['memory_cache']['memory_usage_mb'],
            'lru_eviction_working': final_stats['memory_cache']['size'] <= 50
        }
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests de cache"""
        print("🚀 Tests Complets du Cache Intelligent")
        print("=" * 60)
        
        test_functions = [
            self.test_basic_caching,
            self.test_memory_vs_disk_cache,
            self.test_cache_with_different_data_types,
            self.test_cache_invalidation,
            self.test_specialized_caches,
            self.test_cache_memory_usage
        ]
        
        results = {}
        total_speedup = 0
        speedup_count = 0
        
        for test_func in test_functions:
            try:
                result = test_func()
                results[result['test_name']] = result
                
                # Extract speedup information
                if 'speedup' in result:
                    speedup = result['speedup']
                    if speedup != float('inf') and speedup > 0:
                        total_speedup += speedup
                        speedup_count += 1
                        print(f"✅ {result['test_name']}: {speedup:.1f}x speedup")
                    else:
                        print(f"✅ {result['test_name']}: Completed")
                elif 'avg_speedup' in result:
                    speedup = result['avg_speedup']
                    if speedup != float('inf') and speedup > 0:
                        total_speedup += speedup
                        speedup_count += 1
                        print(f"✅ {result['test_name']}: {speedup:.1f}x avg speedup")
                    else:
                        print(f"✅ {result['test_name']}: Completed")
                else:
                    print(f"✅ {result['test_name']}: Completed")
                
            except Exception as e:
                print(f"❌ Erreur dans {test_func.__name__}: {e}")
                results[test_func.__name__] = {'error': str(e)}
        
        # Calculate overall performance improvement
        avg_speedup = total_speedup / speedup_count if speedup_count > 0 else 0
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"  Tests réussis: {len([r for r in results.values() if 'error' not in r])}/{len(results)}")
        print(f"  Accélération moyenne: {avg_speedup:.1f}x")
        print(f"  Gain de temps moyen: {((avg_speedup - 1) / avg_speedup * 100):.1f}%")
        
        # Global cache stats
        global_stats = get_global_cache_stats()
        print(f"  Cache global - Hit rate: {global_stats['overall_stats']['overall_hit_rate']:.1%}")
        print(f"  Cache global - Mémoire: {global_stats['memory_cache']['memory_usage_mb']:.1f} MB")
        
        # Save results
        self._save_test_results(results, avg_speedup)
        
        return results
    
    def _save_test_results(self, results: Dict[str, Any], avg_speedup: float):
        """Sauvegarde les résultats des tests"""
        from datetime import datetime
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'timestamp': timestamp,
            'average_speedup': avg_speedup,
            'time_saved_percentage': ((avg_speedup - 1) / avg_speedup * 100) if avg_speedup > 0 else 0,
            'global_cache_stats': get_global_cache_stats(),
            'detailed_results': results
        }
        
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/intelligent_cache_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📁 Résultats sauvegardés: {filename}")


def main():
    """Fonction principale"""
    tester = CachePerformanceTester()
    results = tester.run_comprehensive_tests()
    
    return results


if __name__ == "__main__":
    main()