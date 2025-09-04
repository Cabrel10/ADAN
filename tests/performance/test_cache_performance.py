"""
Benchmark de performance pour le système de cache intelligent.

Ce module implémente des tests de performance pour évaluer l'efficacité
du système de cache intelligent, notamment les ratios de succès/échec.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pytest

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au chemin pour les imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import du module de cache intelligent
from adan_trading_bot.utils.intelligent_cache import IntelligentCache

# Constantes pour les tests
CACHE_SIZES = [10, 50, 100]  # Tailles de cache réduites pour des tests plus rapides
TEST_ITERATIONS = 10  # Nombre réduit d'itérations pour les tests de charge


class TestCachePerformance:
    """Classe de tests de performance pour le système de cache."""

    @pytest.fixture(scope="class")
    def sample_data(self) -> Dict[str, Any]:
        """Génère des données de test pour les benchmarks."""
        return {
            'small_array': np.random.rand(10),
            'medium_array': np.random.rand(100, 5),
            'large_array': np.random.rand(1000, 10, 10),
        }

    def test_cache_hit_miss_ratio(self):
        """Teste le ratio de succès/échec du cache avec des types de données simples."""
        for size in CACHE_SIZES:
            # Utiliser un répertoire de cache temporaire
            cache_dir = f"/tmp/cache_test_{size}"

            # Créer une nouvelle instance de cache pour chaque taille
            cache = IntelligentCache(
                memory_cache_size=size,
                disk_cache_dir=cache_dir,
                disk_cache_max_age_hours=1  # Court TTL pour les tests
            )

            logger.info(f"\n=== Test avec taille de cache: {size} ===")

            # Remplir le cache avec des valeurs simples
            logger.info(f"Remplissage du cache avec {size} éléments...")
            for i in range(size):
                # Utiliser des types de données simples (pas de numpy/pandas)
                key = f'key_{i}'
                value = {'data': f'value_{i}', 'timestamp': time.time()}
                cache.put('test_func', (key,), {}, value)

            # Réinitialiser les compteurs avant le test
            cache.memory_hits = 0
            cache.disk_hits = 0
            cache.total_misses = 0

            # Phase 1: Lecture des clés existantes (devrait être des hits mémoire)
            logger.info("Phase 1: Lecture des clés existantes...")
            start_time = time.time()

            for i in range(size):
                key = f'key_{i}'
                result = cache.get('test_func', (key,), {})
                assert result is not None, f"Clé manquante: {key}"
                assert result['data'] == f'value_{i}'

            # Phase 2: Lecture de clés inconnues (devrait être des misses)
            logger.info("Phase 2: Lecture de clés inconnues...")
            for i in range(size, size * 2):
                key = f'unknown_key_{i}'
                result = cache.get('test_func', (key,), {})
                assert result is None, f"Clé inattendue trouvée: {key}"

            end_time = time.time()

            # Afficher les résultats
            total_requests = size * 2  # size hits + size misses
            hit_rate = (cache.memory_hits + cache.disk_hits) / total_requests * 100

            logger.info(f"Résultats pour taille de cache {size}:")
            logger.info(f"- Temps total: {end_time - start_time:.4f} secondes")
            logger.info(f"- Requêtes totales: {total_requests}")
            logger.info(f"- Hits mémoire: {cache.memory_hits}")
            logger.info(f"- Hits disque: {cache.disk_hits}")
            logger.info(f"- Miss totaux: {cache.total_misses}")
            logger.info(f"- Taux de succès: {hit_rate:.2f}%")

            # Vérifications
            assert cache.memory_hits == size, f"Attendu {size} hits mémoire, obtenu {cache.memory_hits}"
            assert cache.disk_hits == 0, f"Attendu 0 hits disque, obtenu {cache.disk_hits}"
            assert cache.total_misses == size, f"Attendu {size} misses, obtenu {cache.total_misses}"

    def test_cache_performance_under_load(self):
        """Teste les performances du cache sous charge avec des données simples."""
        # Utiliser un répertoire de cache temporaire
        cache_dir = "/tmp/cache_perf_test"

        # Créer une instance de cache avec une taille raisonnable
        cache = IntelligentCache(
            memory_cache_size=1000,
            disk_cache_dir=cache_dir,
            disk_cache_max_age_hours=1  # Court TTL pour les tests
        )

        # Données de test simples (pas de numpy/pandas)
        test_data = {
            'small_data': {'value': 42, 'name': 'test'},
            'medium_data': {'items': list(range(100)), 'active': True},
            'large_data': {'data': {str(i): i*2 for i in range(1000)}}
        }
        data_keys = list(test_data.keys())

        logger.info("\n=== Début du test de charge ===")
        logger.info(f"Configuration: {len(data_keys)} clés, {TEST_ITERATIONS} itérations")

        # Réinitialiser les compteurs avant le test
        cache.memory_hits = 0
        cache_disk_hits_before = len(list(Path(cache_dir).glob('*'))) if Path(cache_dir).exists() else 0

        # Phase 1: Mise en cache initiale
        logger.info("Phase 1: Mise en cache initiale...")
        start_time = time.time()

        for i in range(TEST_ITERATIONS):
            for key in data_keys:
                # Ajouter un identifiant unique pour chaque itération
                value = {**test_data[key], 'iteration': i}
                cache.put('test_func', (key,), {}, value, persist_to_disk=True)

        # Phase 2: Lecture des données mises en cache
        logger.info("Phase 2: Lecture des données mises en cache...")
        read_start_time = time.time()

        for _ in range(TEST_ITERATIONS * 2):
            for key in data_keys:
                result = cache.get('test_func', (key,), {})
                assert result is not None, f"Donnée manquante pour la clé: {key}"

        end_time = time.time()

        # Calculer les statistiques
        total_requests = len(data_keys) * TEST_ITERATIONS * 2  # 2x plus de lectures que d'écritures
        total_hits = cache.memory_hits + cache.disk_hits
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        # Vérifier le nombre d'entrées sur disque
        cache_disk_hits_after = len(list(Path(cache_dir).glob('*'))) if Path(cache_dir).exists() else 0
        disk_entries = cache_disk_hits_after - cache_disk_hits_before

        # Afficher les résultats détaillés
        logger.info("\n=== Résultats du test de charge ===")
        logger.info(f"- Durée totale: {end_time - start_time:.2f} secondes")
        logger.info(f"- Durée de lecture: {end_time - read_start_time:.2f} secondes")
        logger.info(f"- Requêtes totales: {total_requests}")
        logger.info(f"- Hits mémoire: {cache.memory_hits} ({(cache.memory_hits/total_requests*100):.1f}%)")
        logger.info(f"- Hits disque: {cache.disk_hits} ({(cache.disk_hits/total_requests*100):.1f}%)")
        logger.info(f"- Miss totaux: {cache.total_misses} ({(cache.total_misses/total_requests*100):.1f}%)")
        logger.info(f"- Taux de succès: {hit_rate:.2f}%")
        logger.info(f"- Entrées sur disque: {disk_entries}")

        # Nettoyer le répertoire de cache temporaire
        import shutil
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)

        # Vérifications finales
        assert hit_rate > 90.0, f"Le taux de succès du cache est trop bas: {hit_rate:.2f}%"
        assert cache.memory_hits > 0, "Aucun hit mémoire détecté"


if __name__ == "__main__":
    # Exécution des tests directement
    test = TestCachePerformance()
    print("=== Début des tests de performance du cache ===")

    # Exécuter le test de ratio hit/miss
    print("\n=== Test de ratio hit/miss ===")
    test.test_cache_hit_miss_ratio()

    # Exécuter le test de charge
    print("\n=== Test de charge ===")
    test.test_cache_performance_under_load(test.sample_data())
