"""
Tests unitaires pour le module preprocessing_cache.py
"""

"""
Tests unitaires pour le module preprocessing_cache.py
"""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from adan_trading_bot.data_processing.preprocessing_cache import (
    PreprocessingCache,
    get_global_cache,
    cached_function,
    _global_cache
)


class TestPreprocessingCache(unittest.TestCase):
    """Tests pour la classe PreprocessingCache"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.cache = PreprocessingCache(
            cache_dir=os.path.join(self.temp_dir, 'cache'),
            memory_cache_size=10,
            verbose=0
        )
        
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Supprimer le répertoire temporaire
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test de l'initialisation du cache"""
        self.assertIsNotNone(self.cache)
        self.assertEqual(self.cache.memory_cache_size, 10)
        self.assertTrue(Path(self.temp_dir, 'cache').exists())
    
    def test_get_data_hash(self):
        """Test de la génération de hash pour différents types de données"""
        # Test avec un DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        hash1 = self.cache._get_data_hash(df)
        hash2 = self.cache._get_data_hash(df.copy())
        self.assertEqual(hash1, hash2)  # Même contenu, même hash
        
        # Modifier légèrement le DataFrame
        df2 = df.copy()
        df2.iloc[0, 0] = 10
        hash3 = self.cache._get_data_hash(df2)
        self.assertNotEqual(hash1, hash3)  # Contenu différent, hash différent
        
        # Test avec un tableau numpy
        arr = np.array([1, 2, 3])
        hash4 = self.cache._get_data_hash(arr)
        hash5 = self.cache._get_data_hash(arr.copy())
        self.assertEqual(hash4, hash5)
    
    def test_cache_operations(self):
        """Test des opérations de base du cache"""
        # Fonction de test
        def expensive_operation(x, multiplier=1):
            time.sleep(0.1)  # Simulation d'une opération coûteuse
            return x * multiplier
        
        # Premier appel - doit calculer et mettre en cache
        start_time = time.time()
        result1 = self.cache.get(expensive_operation, 5, multiplier=2)
        elapsed1 = time.time() - start_time
        self.assertEqual(result1, 10)
        
        # Deuxième appel avec les mêmes arguments - doit utiliser le cache
        start_time = time.time()
        result2 = self.cache.get(expensive_operation, 5, multiplier=2)
        elapsed2 = time.time() - start_time
        self.assertEqual(result2, 10)
        
        # Le deuxième appel doit être beaucoup plus rapide (utilisation du cache)
        self.assertLess(elapsed2, elapsed1 * 0.5)
        
        # Appel avec des arguments différents - doit recalculer
        result3 = self.cache.get(expensive_operation, 5, multiplier=3)
        self.assertEqual(result3, 15)
    
    def test_disk_cache(self):
        """Test de la persistance du cache sur disque"""
        # Créer un cache avec un répertoire spécifique
        cache_dir = os.path.join(self.temp_dir, 'test_disk_cache')
        cache = PreprocessingCache(cache_dir=cache_dir, verbose=0)
        
        # Fonction de test
        def test_func(x):
            return x * 2
        
        # Mettre en cache un résultat
        result1 = cache.get(test_func, 5)
        self.assertEqual(result1, 10)
        
        # Créer une nouvelle instance de cache avec le même répertoire
        cache2 = PreprocessingCache(cache_dir=cache_dir, verbose=0)
        
        # Le résultat devrait être chargé depuis le cache disque
        result2 = cache2.get(test_func, 5)
        self.assertEqual(result2, 10)
    
    def test_memory_cache_eviction(self):
        """Test de l'éviction du cache mémoire (LRU)."""
        cache = PreprocessingCache(memory_cache_size=2, verbose=0)
        
        # Fonction de test
        def test_func(x):
            return x
            
        # Générer des clés de cache avec la signature correcte
        key1 = cache._get_cache_key(test_func, (1,), {})
        key2 = cache._get_cache_key(test_func, (2,), {})
        key3 = cache._get_cache_key(test_func, (3,), {})
        
        # Utiliser la méthode _update_memory_cache pour gérer correctement l'éviction LRU
        cache._update_memory_cache(key1, 1)
        cache._update_memory_cache(key2, 2)
        
        # Ajouter un 3ème élément - le premier devrait être évincé
        cache._update_memory_cache(key3, 3)
        
        # Vérifications
        self.assertEqual(len(cache.memory_cache), 2)
        self.assertNotIn(key1, cache.memory_cache)
        self.assertIn(key2, cache.memory_cache)
        self.assertIn(key3, cache.memory_cache)
    
    def test_clear_cache(self):
        """Test de la suppression du cache"""
        # Ajouter des éléments au cache
        self.cache.get(lambda x: x, 1)
        self.cache.get(lambda x: x, 2)
        
        # Vérifier que le cache n'est pas vide
        self.assertTrue(self.cache.memory_cache)
        
        # Vider le cache
        self.cache.clear()
        
        # Vérifier que le cache est vide
        self.assertFalse(self.cache.memory_cache)
        
        # Vérifier que le répertoire de cache existe toujours
        self.assertTrue(Path(self.temp_dir, 'cache').exists())


class TestCachedFunctionDecorator(unittest.TestCase):
    """Tests pour le décorateur cached_function"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = PreprocessingCache(
            cache_dir=os.path.join(self.temp_dir, 'cache'),
            verbose=0
        )
        
        # Décorer une fonction de test
        @cached_function(cache=self.cache)
        def test_function(x, multiplier=1):
            return x * multiplier
            
        self.test_function = test_function
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cached_function(self):
        """Test du décorateur cached_function"""
        # Premier appel - doit calculer et mettre en cache
        result1 = self.test_function(5, multiplier=2)
        self.assertEqual(result1, 10)
        
        # Deuxième appel avec les mêmes arguments - doit utiliser le cache
        with patch('adan_trading_bot.data_processing.preprocessing_cache.PreprocessingCache.get') as mock_get:
            mock_get.return_value = 10
            result2 = self.test_function(5, multiplier=2)
            mock_get.assert_called_once()
            self.assertEqual(result2, 10)
        
        # Appel avec des arguments différents - doit recalculer
        result3 = self.test_function(5, multiplier=3)
        self.assertEqual(result3, 15)


class TestGlobalCache(unittest.TestCase):
    """Tests pour le cache global"""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Réinitialiser le cache global
        global _global_cache
        _global_cache = None
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Réinitialiser le cache global
        global _global_cache
        _global_cache = None
    
    def test_get_global_cache(self):
        """Test de la fonction get_global_cache."""
        # Premier appel - doit créer une nouvelle instance
        cache1 = get_global_cache()
        self.assertIsNotNone(cache1)
        
        # Deuxième appel - doit retourner la même instance
        cache2 = get_global_cache()
        self.assertIs(cache2, cache1)
    
    def test_global_cache_with_custom_dir(self):
        """Test du cache global avec un répertoire personnalisé."""
        cache_dir = os.path.join(self.temp_dir, 'global_cache')
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(cache_dir, exist_ok=True)
        
        # Obtenir le cache avec le répertoire personnalisé
        cache = get_global_cache(cache_dir=cache_dir)
        
        # Vérifications
        self.assertTrue(os.path.exists(cache_dir))
        result = cache.get(lambda x: x * 2, 5)
        self.assertEqual(result, 10)


if __name__ == '__main__':
    unittest.main()
